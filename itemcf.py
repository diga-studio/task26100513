"""Incremental ItemCF neighbor index with adjusted cosine similarity.

Adjusted cosine: c(u, i) = rating(u, i) - mean(ratings of u). Similarity of
items i, j is sum_u c(u,i)*c(u,j) / sqrt(sum_u c(u,i)^2 * sum_u c(u,j)^2),
where every sum runs over all users that rated the item (not only co-raters).
Scores are quantized with round(score, 10); only positive scores are kept.

All centered values are rationals with denominator n = |ratings of u|. To keep
updates exact (so that incremental state always matches a fresh rebuild
bit-for-bit), every accumulated quantity is stored as an integer scaled by
SCALE^2, where SCALE is a common multiple of all seen user-rating counts.
The scale cancels in the similarity ratio, so scores are computed directly
from the integer accumulators.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter

_SCALE = math.lcm(*range(1, 31))  # covers the supported <=30 ratings per user


class ItemCFIndex:
    """In-memory, incrementally updated item-item neighbor index."""

    TOP_K = 50

    def __init__(self) -> None:
        # user_id -> {item_id: rating}
        self._ratings: dict[int, dict[int, int]] = {}
        # item_id -> integer norm accumulator (sum of c(u,i)^2 * SCALE^2)
        self._norm: dict[int, int] = {}
        # item_id -> {other_item_id: [dot_accumulator, user_count]}; the list
        # object is shared between both directions of a pair.
        self._pairs: dict[int, dict[int, list]] = {}
        # item_id -> {other_item_id: quantized score} (positive scores only)
        self._nbr_score: dict[int, dict[int, float]] = {}
        # item_id -> sorted list of (-score, other_item_id)
        self._nbr_list: dict[int, list[tuple[float, int]]] = {}
        self._scale = _SCALE
        # instrumentation: lets tests verify update locality and query cost
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def build(self, ratings) -> None:
        """Build the index from (user_id, item_id, rating) records."""
        self.__init__()
        by_user: dict[int, dict[int, int]] = {}
        for user_id, item_id, rating in ratings:
            by_user.setdefault(user_id, {})[item_id] = rating
        for user_id, items in by_user.items():
            self._ratings[user_id] = items
            self._add_user_contrib(items)
        seen = set()
        for i, row in self._pairs.items():
            for j in row:
                if (i, j) not in seen:
                    seen.add((j, i))
                    self._refresh_pair(i, j)

    def apply(self, changes) -> None:
        """Apply (user_id, item_id, rating|None) upserts/deletes atomically."""
        by_user: dict[int, dict[int, int | None]] = {}
        for user_id, item_id, rating in changes:
            by_user.setdefault(user_id, {})[item_id] = rating

        changed_items: set[int] = set()
        affected_pairs: set[tuple[int, int]] = set()

        for user_id, mods in by_user.items():
            old = self._ratings.get(user_id)
            if old:
                self._remove_user_contrib(old, affected_pairs)
            new = dict(old) if old else {}
            for item_id, rating in mods.items():
                if rating is None:
                    new.pop(item_id, None)
                else:
                    new[item_id] = rating
            if new:
                self._ratings[user_id] = new
                self._add_user_contrib(new)
            else:
                self._ratings.pop(user_id, None)
            changed_items.update(new)
            if old:
                changed_items.update(old)

        # A pair's score changes iff its dot changed (co-rating user changed)
        # or an endpoint norm changed. Both cases are exactly the pairs
        # incident to items of changed users, plus pairs that died.
        for item_id in changed_items:
            row = self._pairs.get(item_id)
            if row:
                for other in row:
                    if item_id < other:
                        affected_pairs.add((item_id, other))
                    else:
                        affected_pairs.add((other, item_id))
        for i, j in affected_pairs:
            self._refresh_pair(i, j)

    def neighbors(self, item_id: int) -> list[tuple[int, float]]:
        """Up to TOP_K (neighbor_id, score), score desc then id asc."""
        lst = self._nbr_list.get(item_id)
        if not lst:
            return []
        return [(j, -neg) for neg, j in lst[: self.TOP_K]]

    # ------------------------------------------------------------------ #
    # contribution accumulation (exact integer arithmetic)
    # ------------------------------------------------------------------ #

    def _user_terms(self, items: dict[int, int]) -> list[tuple[int, int]]:
        """(item, k) with k = c(u,item) * SCALE, an exact integer."""
        n = len(items)
        if self._scale % n:
            self._rescale(n)
        factor = self._scale // n
        total = sum(items.values())
        return [(i, (r * n - total) * factor) for i, r in items.items()]

    def _rescale(self, n: int) -> None:
        new_scale = math.lcm(self._scale, n)
        factor = (new_scale // self._scale) ** 2
        for item_id in self._norm:
            self._norm[item_id] *= factor
        done = set()
        for row in self._pairs.values():
            for cell in row.values():
                if id(cell) not in done:
                    done.add(id(cell))
                    cell[0] *= factor
        self._scale = new_scale

    def _add_user_contrib(self, items: dict[int, int]) -> None:
        terms = self._user_terms(items)
        for i, ki in terms:
            self._norm[i] = self._norm.get(i, 0) + ki * ki
        for a in range(len(terms)):
            i, ki = terms[a]
            row = self._pairs.setdefault(i, {})
            for j, kj in terms[a + 1 :]:
                cell = row.get(j)
                if cell is None:
                    cell = [0, 0]
                    row[j] = cell
                    self._pairs.setdefault(j, {})[i] = cell
                cell[0] += ki * kj
                cell[1] += 1
        self.stats["user_contrib_updates"] += 1

    def _remove_user_contrib(
        self, items: dict[int, int], affected_pairs: set[tuple[int, int]]
    ) -> None:
        terms = self._user_terms(items)
        for i, ki in terms:
            self._norm[i] -= ki * ki
        for a in range(len(terms)):
            i, ki = terms[a]
            row = self._pairs[i]
            for j, kj in terms[a + 1 :]:
                cell = row[j]
                cell[0] -= ki * kj
                cell[1] -= 1
                if cell[1] == 0:
                    del row[j]
                    del self._pairs[j][i]
                    affected_pairs.add((i, j) if i < j else (j, i))
        self.stats["user_contrib_updates"] += 1

    # ------------------------------------------------------------------ #
    # neighbor maintenance
    # ------------------------------------------------------------------ #

    def _pair_score(self, i: int, j: int):
        cell = self._pairs.get(i, {}).get(j)
        if cell is None:
            return None
        ni = self._norm.get(i, 0)
        nj = self._norm.get(j, 0)
        if ni <= 0 or nj <= 0 or cell[0] <= 0:
            return None
        score = round(cell[0] / math.sqrt(ni * nj), 10)
        return score if score > 0 else None

    def _refresh_pair(self, i: int, j: int) -> None:
        score = self._pair_score(i, j)
        self._set_neighbor(i, j, score)
        self._set_neighbor(j, i, score)
        self.stats["pair_refresh"] += 1

    def _set_neighbor(self, i: int, j: int, score) -> None:
        scores = self._nbr_score.get(i)
        old = scores.get(j) if scores is not None else None
        if old is not None:
            if score is not None and score == old:
                return
            lst = self._nbr_list[i]
            del lst[bisect.bisect_left(lst, (-old, j))]
            del scores[j]
            self.stats["neighbor_updates"] += 1
        if score is not None:
            if scores is None:
                scores = self._nbr_score[i] = {}
                self._nbr_list[i] = []
            scores[j] = score
            bisect.insort(self._nbr_list[i], (-score, j))
            self.stats["neighbor_updates"] += 1
