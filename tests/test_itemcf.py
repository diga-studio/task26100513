"""Tests for itemcf.ItemCFIndex."""

import math
import random
import resource
from collections import defaultdict

import pytest

from itemcf import ItemCFIndex


def reference_neighbors(ratings, top_k=50):
    """Independent from-scratch computation.

    ratings: dict[(user_id, item_id)] -> rating
    returns: dict[item_id] -> list of (item_id, score)
    """
    user_items = defaultdict(dict)
    for (user_id, item_id), rating in ratings.items():
        user_items[user_id][item_id] = rating
    num = defaultdict(float)
    norm = defaultdict(float)
    for items in user_items.values():
        mean = sum(items.values()) / len(items)
        centered = {item: rating - mean for item, rating in items.items()}
        for item, c in centered.items():
            norm[item] += c * c
        pairs = sorted(centered.items())
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                num[(pairs[a][0], pairs[b][0])] += pairs[a][1] * pairs[b][1]
    sims = defaultdict(dict)
    for (item_a, item_b), n in num.items():
        na, nb = norm.get(item_a, 0.0), norm.get(item_b, 0.0)
        score = 0.0 if na == 0.0 or nb == 0.0 else round(n / math.sqrt(na * nb), 10)
        sims[item_a][item_b] = score
        sims[item_b][item_a] = score
    result = {}
    for item, entries in sims.items():
        positive = [(other, s) for other, s in entries.items() if s > 0]
        positive.sort(key=lambda entry: (-entry[1], entry[0]))
        result[item] = positive[:top_k]
    return result


def assert_all_items_match(idx, ratings, ever_seen_items):
    ref = reference_neighbors(ratings)
    for item in ever_seen_items:
        assert idx.neighbors(item) == ref.get(item, []), f"item {item} mismatch"


def test_build_basic_and_unknown_item():
    idx = ItemCFIndex()
    idx.build([(1, 10, 5), (1, 11, 1), (2, 10, 4), (2, 11, 2), (2, 12, 5)])
    ratings = {(1, 10): 5, (1, 11): 1, (2, 10): 4, (2, 11): 2, (2, 12): 5}
    assert_all_items_match(idx, ratings, {10, 11, 12})
    assert idx.neighbors(999) == []
    # self never appears in own neighbor list
    for item in (10, 11, 12):
        assert all(other != item for other, _ in idx.neighbors(item))


def test_zero_norm_item_has_no_neighbors():
    idx = ItemCFIndex()
    # user 1 rates only item 1 -> centered value 0 -> zero norm
    # user 2 rates items 1 and 2 identically -> both centered 0 for user 2
    idx.build([(1, 1, 4), (2, 1, 3), (2, 2, 3), (3, 2, 5), (3, 3, 1)])
    assert idx.neighbors(1) == []
    assert all(other != 1 for other, _ in idx.neighbors(2))
    assert all(other != 1 for other, _ in idx.neighbors(3))


def test_random_batches_match_reference():
    rng = random.Random(20261005)
    idx = ItemCFIndex()
    ratings = {}
    ever_items = set()

    initial = {}
    for user in range(15):
        for item in rng.sample(range(40), rng.randint(1, 8)):
            initial[(user, item)] = rng.randint(1, 5)
    ratings.update(initial)
    ever_items.update(item for _, item in ratings)
    idx.build([(u, i, r) for (u, i), r in ratings.items()])
    assert_all_items_match(idx, ratings, ever_items)

    for _ in range(150):
        changes = []
        users = rng.sample(range(20), rng.randint(1, 6))  # includes brand-new users
        for user in users:
            current = [i for (u, i) in ratings if u == user]
            ops = rng.randint(1, 4)
            keys = set()
            for _ in range(ops):
                if current and rng.random() < 0.6:
                    item = rng.choice(current)
                else:
                    item = rng.randrange(50)  # includes brand-new items
                if item in keys:
                    continue
                keys.add(item)
                if (user, item) in ratings and rng.random() < 0.4:
                    rating = None
                else:
                    rating = rng.randint(1, 5)
                changes.append((user, item, rating))
        idx.apply(changes)
        for user, item, rating in changes:
            ever_items.add(item)
            if rating is None:
                ratings.pop((user, item), None)
            else:
                ratings[(user, item)] = rating
        assert_all_items_match(idx, ratings, ever_items)


def test_delete_changes_user_mean_and_zero_norm():
    idx = ItemCFIndex()
    idx.build([(1, 1, 5), (1, 2, 1), (2, 1, 4), (2, 2, 2), (2, 3, 5)])
    # delete user 1's second rating: user 1's mean changes, item 1's centered
    # value for user 1 becomes 0 (single remaining rating)
    idx.apply([(1, 2, None)])
    ratings = {(1, 1): 5, (2, 1): 4, (2, 2): 2, (2, 3): 5}
    assert_all_items_match(idx, ratings, {1, 2, 3})
    # delete everything of user 2 as well: item 1 zero-norm, item 2/3 gone
    idx.apply([(2, 1, None), (2, 2, None), (2, 3, None)])
    ratings = {(1, 1): 5}
    assert_all_items_match(idx, ratings, {1, 2, 3})
    assert idx.neighbors(1) == []
    assert idx.neighbors(2) == []


def _build_tie_scenario():
    """Item 1 co-occurs with items 100..150 (51 candidates), all tied."""
    idx = ItemCFIndex()
    ratings = []
    for k in range(51):
        user = 1000 + k
        ratings.append((user, 1, 5))
        ratings.append((user, 100 + k, 5))
        ratings.append((user, 1000 + k, 1))  # unique anchor, negative centered
    idx.build(ratings)
    return idx


def test_tie_break_by_item_id_and_top50_cut():
    idx = _build_tie_scenario()
    top = idx.neighbors(1)
    assert len(top) == 50
    scores = {score for _, score in top}
    assert len(scores) == 1  # perfect tie
    assert [item for item, _ in top] == list(range(100, 150))  # ascending ids
    assert all(item != 150 for item, _ in top)  # 51st candidate excluded


def test_norm_change_promotes_51st_candidate():
    idx = _build_tie_scenario()
    before = idx.neighbors(1)
    assert [item for item, _ in before][-1] == 149

    # Other users (not touching item 1) inflate the norm of item 149.
    changes = []
    for k in range(5):
        user = 5000 + k
        changes.append((user, 149, 5))
        changes.append((user, 2000 + k, 1))
    idx.apply(changes)

    after = idx.neighbors(1)
    ids = [item for item, _ in after]
    assert 150 in ids  # former 51st candidate promoted
    assert 149 not in ids  # demoted by its increased norm
    assert len(after) == 50
    # item 1 itself was never directly updated, yet its board changed
    assert after != before
    # item 150 was not directly updated either, but now lists item 1
    assert any(other == 1 for other, _ in idx.neighbors(150))
    # demoted item 149 still sees item 1, but with a lower score than item 150
    score_149 = dict(idx.neighbors(149))[1]
    score_150 = dict(idx.neighbors(150))[1]
    assert score_149 < score_150


def _make_grouped_ratings(num_users=20_000, num_items=10_000, group_size=100,
                          per_user=10, seed=7):
    """Users rate only inside their own item group -> no cross-group pairs."""
    rng = random.Random(seed)
    ratings = []
    groups = [list(range(g, g + group_size)) for g in range(0, num_items, group_size)]
    for user in range(num_users):
        group = groups[user % len(groups)]
        for item in rng.sample(group, per_user):
            ratings.append((user, item, rng.randint(1, 5)))
    return ratings, groups


def test_large_sparse_regression_scoped_updates():
    ratings, groups = _make_grouped_ratings()
    assert len(ratings) == 200_000

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    idx = ItemCFIndex()
    idx.build(ratings)

    mirror = {(u, i): r for u, i, r in ratings}
    rng = random.Random(99)

    # batch touches only users of group 0 (users 0, 100, 200, ... are group 0)
    group0_items = set(groups[0])
    group0_users = [u for u in range(0, 20_000, 100)][:10]
    changes = []
    for user in group0_users:
        user_items = [i for (u, i) in mirror if u == user]
        for item in rng.sample(user_items, 4):
            changes.append((user, item, rng.randint(1, 5)))
        changes.append((user, rng.choice(sorted(group0_items)), None))

    stats_before = dict(idx.stats)
    idx.apply(changes)
    for user, item, rating in changes:
        if rating is None:
            mirror.pop((user, item), None)
        else:
            mirror[(user, item)] = rating

    # --- access-count proof: work stayed inside group 0 ---
    assert idx.last_dirty_items <= group0_items
    total_pairs = len(idx._pair_cnt)
    reevaluated = idx.stats["pairs_reevaluated"] - stats_before["pairs_reevaluated"]
    scanned = idx.stats["top_scanned_entries"] - stats_before["top_scanned_entries"]
    assert reevaluated < total_pairs / 50  # 100 groups; touched << 2% of pairs
    assert scanned < total_pairs / 50

    # --- neighbors() does O(result) work, no candidate scan ---
    probe = sorted(group0_items)[0]
    reads_before = idx.stats["neighbor_reads"]
    result = idx.neighbors(probe)
    assert idx.stats["neighbor_reads"] - reads_before == len(result)
    empty_reads_before = idx.stats["neighbor_reads"]
    assert idx.neighbors(999_999) == []
    assert idx.stats["neighbor_reads"] == empty_reads_before

    # --- correctness against from-scratch reference ---
    ref = reference_neighbors(mirror)
    for item in group0_items:
        assert idx.neighbors(item) == ref.get(item, []), f"group0 item {item}"
    other_items = [g[0] for g in groups[1:]]
    for item in rng.sample(other_items, 50):
        assert idx.neighbors(item) == ref.get(item, []), f"untouched item {item}"

    # --- memory: peak RSS growth for build+apply stays under 1 GiB ---
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rss_after - rss_before < 1 << 30
