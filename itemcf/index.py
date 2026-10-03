"""In-memory incremental ItemCF neighbor index (adjusted cosine similarity)."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

Rating = Tuple[int, int, int]
Change = Tuple[int, int, Optional[int]]
Neighbor = Tuple[int, float]

TOP_K = 50
NDIGITS = 10
# True norms are either exactly 0 or >= (1/30)^2 for integer ratings with
# at most 30 ratings per user; anything below this is float residue.
_NORM_EPS = 1e-9


class ItemCFIndex:
    """Incremental ItemCF neighbor index.

    All state lives in memory. Methods are plain synchronous calls.
    """

    def __init__(self) -> None:
        # user_id -> {item_id: rating}
        self._user_ratings: Dict[int, Dict[int, int]] = {}
        # item_id -> {user_id: rating}
        self._item_users: Dict[int, Dict[int, int]] = {}
        # item_id -> sum over its users of centered rating squared
        self._norm: Dict[int, float] = {}
        # (item_a, item_b) with a < b -> sum of centered-rating products
        self._pair_num: Dict[Tuple[int, int], float] = {}
        # (item_a, item_b) -> number of users currently rating both
        self._pair_cnt: Dict[Tuple[int, int], int] = {}
        # item_id -> {other_item_id: quantized similarity}; one entry per
        # live co-occurrence pair, in both directions
        self._sims: Dict[int, Dict[int, float]] = {}
        # item_id -> precomputed top list of (item_id, score)
        self._top: Dict[int, List[Neighbor]] = {}
        # instrumentation used by tests / diagnostics
        self.stats: Dict[str, int] = {
            "pair_updates": 0,
            "pairs_reevaluated": 0,
            "top_rebuilds": 0,
            "top_scanned_entries": 0,
            "neighbor_reads": 0,
        }
        self.last_dirty_items: frozenset = frozenset()

    def build(self, ratings: Iterable[Rating]) -> None:
        """Rebuild the index from scratch given (user_id, item_id, rating)."""
        self.__init__()
        per_user: Dict[int, Dict[int, int]] = {}
        for user_id, item_id, rating in ratings:
            per_user.setdefault(user_id, {})[item_id] = rating
            self._item_users.setdefault(item_id, {})[user_id] = rating
        self._user_ratings = per_user

        pair_num = self._pair_num
        pair_cnt = self._pair_cnt
        norm = self._norm
        for items in per_user.values():
            mean = sum(items.values()) / len(items)
            centered = [(item, rating - mean) for item, rating in items.items()]
            centered.sort()
            for item, c in centered:
                norm[item] = norm.get(item, 0.0) + c * c
            for a in range(len(centered)):
                item_a, c_a = centered[a]
                for b in range(a + 1, len(centered)):
                    item_b, c_b = centered[b]
                    key = (item_a, item_b)
                    pair_num[key] = pair_num.get(key, 0.0) + c_a * c_b
                    pair_cnt[key] = pair_cnt.get(key, 0) + 1
        self.stats["pair_updates"] += len(pair_num)

        for (item_a, item_b), num in pair_num.items():
            score = self._score(num, norm.get(item_a, 0.0), norm.get(item_b, 0.0))
            self._sims.setdefault(item_a, {})[item_b] = score
            self._sims.setdefault(item_b, {})[item_a] = score
        for item_id in self._sims:
            self._rebuild_top(item_id)

    def apply(self, changes: Iterable[Change]) -> None:
        """Apply a batch of (user_id, item_id, rating|None) changes."""
        by_user: Dict[int, Dict[int, Optional[int]]] = {}
        for user_id, item_id, rating in changes:
            by_user.setdefault(user_id, {})[item_id] = rating

        touched_pairs: set = set()
        norm_changed: set = set()
        for user_id, mods in by_user.items():
            self._apply_user(user_id, mods, touched_pairs, norm_changed)

        # Re-evaluate every pair whose numerator changed, plus every live
        # pair incident to an item whose norm changed.
        pairs_to_eval = set(touched_pairs)
        for item_id in norm_changed:
            for other in self._sims.get(item_id, ()):
                pairs_to_eval.add(
                    (item_id, other) if item_id < other else (other, item_id)
                )

        dirty: set = set()
        self.stats["pairs_reevaluated"] += len(pairs_to_eval)
        for item_a, item_b in pairs_to_eval:
            sims_a = self._sims.get(item_a)
            old = sims_a.get(item_b) if sims_a else None
            if not self._pair_cnt.get((item_a, item_b)):
                if old is not None:
                    del sims_a[item_b]
                    if not sims_a:
                        del self._sims[item_a]
                    sims_b = self._sims[item_b]
                    del sims_b[item_a]
                    if not sims_b:
                        del self._sims[item_b]
                    dirty.add(item_a)
                    dirty.add(item_b)
                continue
            new = self._score(
                self._pair_num.get((item_a, item_b), 0.0),
                self._norm.get(item_a, 0.0),
                self._norm.get(item_b, 0.0),
            )
            if old != new:
                self._sims.setdefault(item_a, {})[item_b] = new
                self._sims.setdefault(item_b, {})[item_a] = new
                dirty.add(item_a)
                dirty.add(item_b)

        for item_id in dirty:
            self._rebuild_top(item_id)
        self.last_dirty_items = frozenset(dirty)

    def neighbors(self, item_id: int) -> List[Neighbor]:
        """Return up to TOP_K (item_id, similarity) entries for item_id."""
        top = self._top.get(item_id)
        if not top:
            return []
        self.stats["neighbor_reads"] += len(top)
        return list(top)

    @staticmethod
    def _score(num: float, norm_a: float, norm_b: float) -> float:
        if norm_a < _NORM_EPS or norm_b < _NORM_EPS:
            return 0.0
        return round(num / math.sqrt(norm_a * norm_b), NDIGITS)

    def _rebuild_top(self, item_id: int) -> None:
        sims = self._sims.get(item_id)
        self.stats["top_rebuilds"] += 1
        if not sims:
            self._top.pop(item_id, None)
            return
        self.stats["top_scanned_entries"] += len(sims)
        entries = [(other, score) for other, score in sims.items() if score > 0]
        if not entries:
            self._top.pop(item_id, None)
            return
        entries.sort(key=lambda entry: (-entry[1], entry[0]))
        self._top[item_id] = entries[:TOP_K]

    def _apply_user(self, user_id, mods, touched_pairs, norm_changed) -> None:
        old = self._user_ratings.get(user_id)
        old = dict(old) if old else {}
        new = dict(old)
        for item_id, rating in mods.items():
            if rating is None:
                new.pop(item_id, None)
            else:
                new[item_id] = rating

        old_mean = sum(old.values()) / len(old) if old else 0.0
        new_mean = sum(new.values()) / len(new) if new else 0.0
        old_c = {item: rating - old_mean for item, rating in old.items()}
        new_c = {item: rating - new_mean for item, rating in new.items()}

        if new:
            self._user_ratings[user_id] = new
        else:
            self._user_ratings.pop(user_id, None)
        for item_id, rating in new.items():
            self._item_users.setdefault(item_id, {})[user_id] = rating
        emptied_items = []
        for item_id in old:
            if item_id not in new:
                users = self._item_users[item_id]
                del users[user_id]
                if not users:
                    del self._item_users[item_id]
                    emptied_items.append(item_id)

        union = sorted(set(old_c) | set(new_c))
        norm = self._norm
        for item_id in union:
            c_new = new_c.get(item_id, 0.0)
            c_old = old_c.get(item_id, 0.0)
            delta = c_new * c_new - c_old * c_old
            if delta:
                norm[item_id] = norm.get(item_id, 0.0) + delta
                norm_changed.add(item_id)

        for item_id in emptied_items:
            self._norm.pop(item_id, None)
            self._top.pop(item_id, None)

        pair_num = self._pair_num
        pair_cnt = self._pair_cnt
        for a in range(len(union)):
            item_a = union[a]
            ca_new = new_c.get(item_a, 0.0)
            ca_old = old_c.get(item_a, 0.0)
            for b in range(a + 1, len(union)):
                item_b = union[b]
                delta = ca_new * new_c.get(item_b, 0.0) - ca_old * old_c.get(item_b, 0.0)
                cnt_delta = (item_a in new_c and item_b in new_c) - (
                    item_a in old_c and item_b in old_c
                )
                if not delta and not cnt_delta:
                    continue
                key = (item_a, item_b)
                if delta:
                    pair_num[key] = pair_num.get(key, 0.0) + delta
                if cnt_delta:
                    count = pair_cnt.get(key, 0) + cnt_delta
                    if count <= 0:
                        pair_num.pop(key, None)
                        pair_cnt.pop(key, None)
                    else:
                        pair_cnt[key] = count
                touched_pairs.add(key)
                self.stats["pair_updates"] += 1
