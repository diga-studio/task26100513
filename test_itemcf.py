"""Tests for itemcf.ItemCFIndex."""

import math
import random
from fractions import Fraction

import pytest

from itemcf import ItemCFIndex

SCALE = math.lcm(*range(1, 31))


def reference_neighbors(ratings):
    """Independent full recomputation from raw ratings.

    ratings: {user_id: {item_id: rating}}. Accumulates exact rationals with
    Fraction, then converts to the same scaled-integer score expression the
    library uses so float results are comparable bit-for-bit.
    Returns {item_id: [(neighbor_id, score), ...]} for every item.
    """
    sizes = {len(items) for items in ratings.values()}
    scale = SCALE
    for n in sizes:
        scale = math.lcm(scale, n)
    dots = {}
    norms = {}
    items_seen = set()
    for items in ratings.values():
        n = len(items)
        mean = Fraction(sum(items.values()), n)
        centered = {i: Fraction(r) - mean for i, r in items.items()}
        items_seen.update(centered)
        for i, c in centered.items():
            norms[i] = norms.get(i, Fraction(0)) + c * c
        keys = sorted(centered)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                i, j = keys[a], keys[b]
                dots[(i, j)] = dots.get((i, j), Fraction(0)) + centered[i] * centered[j]
    scaled_norm = {i: int(norms.get(i, 0) * scale * scale) for i in items_seen}
    cand = {i: [] for i in items_seen}
    for (i, j), d in dots.items():
        if d <= 0:
            continue
        ni, nj = scaled_norm[i], scaled_norm[j]
        if ni <= 0 or nj <= 0:
            continue
        kd = int(d * scale * scale)
        score = round(kd / math.sqrt(ni * nj), 10)
        if score > 0:
            cand[i].append((-score, j))
            cand[j].append((-score, i))
    out = {}
    for i in items_seen:
        cand[i].sort()
        out[i] = [(j, -neg) for neg, j in cand[i][:50]]
    return out


def check_all(index, truth, item_ids):
    ref = reference_neighbors(truth)
    for item in item_ids:
        assert index.neighbors(item) == ref.get(item, []), f"item {item}"


def random_batch(rng, truth, n_users, n_items):
    users = rng.sample(range(n_users), rng.randint(1, 4))
    batch = []
    for u in users:
        rated = list(truth.get(u, {}))
        ops = rng.randint(1, 3)
        for _ in range(ops):
            choice = rng.random()
            if rated and choice < 0.35:
                batch.append((u, rng.choice(rated), None))  # delete
            elif rated and choice < 0.7:
                batch.append((u, rng.choice(rated), rng.randint(1, 5)))  # modify
            else:
                item = rng.randrange(n_items)
                if item not in truth.get(u, {}):
                    batch.append((u, item, rng.randint(1, 5)))
    # keep batch keys unique and consistent with truth semantics
    seen = {}
    for u, i, r in batch:
        seen[(u, i)] = r
    return [(u, i, r) for (u, i), r in seen.items()]


def apply_to_truth(truth, batch):
    for u, i, r in batch:
        if r is None:
            truth.get(u, {}).pop(i, None)
            if u in truth and not truth[u]:
                del truth[u]
        else:
            truth.setdefault(u, {})[i] = r


def test_randomized_against_reference():
    rng = random.Random(20261003)
    n_users, n_items = 40, 60
    truth = {}
    initial = []
    for u in range(n_users):
        for _ in range(rng.randint(0, 8)):
            item = rng.randrange(n_items)
            if item not in truth.get(u, {}):
                r = rng.randint(1, 5)
                truth.setdefault(u, {})[item] = r
                initial.append((u, item, r))
    index = ItemCFIndex()
    index.build(initial)
    check_all(index, truth, list(range(n_items)) + [10_000])
    for _ in range(300):
        batch = random_batch(rng, truth, n_users, n_items)
        index.apply(batch)
        apply_to_truth(truth, batch)
        check_all(index, truth, range(n_items))


def test_zero_norm_items_have_no_neighbors():
    index = ItemCFIndex()
    # user 1 rates only item 5 -> c(1,5) = 0 -> item 5 has zero norm
    # users 2,3 rate three items so 5 and 6 are positively correlated
    index.build(
        [(1, 5, 3), (2, 5, 5), (2, 6, 5), (2, 7, 1),
         (3, 5, 4), (3, 6, 4), (3, 7, 2)]
    )
    assert [j for j, _ in index.neighbors(5)] == [6]
    # remove all ratings that give item 6 a nonzero norm
    index.apply([(2, 6, None), (3, 6, None)])
    assert index.neighbors(6) == []
    # item 6 must also disappear from item 5's list
    assert all(j != 6 for j, _ in index.neighbors(5))
    # delete every rating of item 5 -> it vanishes everywhere
    index.apply([(1, 5, None), (2, 5, None), (3, 5, None)])
    assert index.neighbors(5) == []
    assert index.neighbors(6) == []


def test_user_mean_shift_on_delete():
    # deleting one rating changes the user's mean, hence every centered value
    index = ItemCFIndex()
    index.build([(1, 1, 5), (1, 2, 1), (1, 3, 4), (2, 1, 4), (2, 2, 2)])
    before = index.neighbors(1)
    index.apply([(1, 3, None)])
    truth = {1: {1: 5, 2: 1}, 2: {1: 4, 2: 2}}
    check_all(index, truth, [1, 2, 3])
    assert index.neighbors(1) != before


T, F, G = 0, 9998, 9999


def build_candidate_index(counts):
    """Item T plus one candidate per count; candidate j gets `count` dedicated
    users rating {T:5, j:5, F:1}, so score(T, j) grows with sqrt(count)."""
    ratings = []
    uid = 0
    for item, m in counts.items():
        for _ in range(m):
            ratings += [(uid, T, 5), (uid, item, 5), (uid, F, 1)]
            uid += 1
    index = ItemCFIndex()
    index.build(ratings)
    return index, uid


def run_promotion_scenario():
    # candidates 1..55 with distinct scores, ranked j=55 best ... j=1 worst
    counts = {j: j for j in range(1, 56)}
    index, next_uid = build_candidate_index(counts)
    board = index.neighbors(T)
    assert len(board) == 50
    assert board[0][0] == 55 and board[1][0] == 54
    assert board[-1][0] == 6  # rank 50
    assert 5 not in [j for j, _ in board]  # rank 51 is just outside

    # 300 brand-new users rate item 55 (and filler G) but NOT item T, in
    # batches of 10 users: norm(55) keeps growing until score(T,55) falls
    # below the old rank 50 and item 55 leaves the board entirely.
    for b in range(30):
        batch = []
        for _ in range(10):
            batch += [(next_uid, 55, 5), (next_uid, G, 1)]
            next_uid += 1
        index.apply(batch)

    board = index.neighbors(T)
    ids = [j for j, _ in board]
    assert len(board) == 50
    assert ids[0] == 54 and ids[1] == 53  # 54 is the new best neighbor
    assert 55 not in ids  # norm inflation pushed the old leader out
    assert ids[-1] == 5  # former rank 51 is promoted into the board
    assert 6 in ids
    # item T was never part of any applied change, yet its board re-ranked
    return index, next_uid


def test_rank51_promotion_and_untouched_item_reboards():
    run_promotion_scenario()


def test_rank51_full_state_matches_reference():
    index, next_uid = run_promotion_scenario()
    truth = {}
    uid = 0
    for j in range(1, 56):
        for _ in range(j):
            truth[uid] = {T: 5, j: 5, F: 1}
            uid += 1
    for extra in range(uid, next_uid):
        truth[extra] = {55: 5, G: 1}
    check_all(index, truth, [T, 1, 5, 6, 54, 55, F, G])


def test_ties_broken_by_item_id():
    # 55 candidates with identical scores -> top 50 are ids 1..50 ascending
    counts = {j: 1 for j in range(1, 56)}
    index, _ = build_candidate_index(counts)
    board = index.neighbors(T)
    assert [j for j, _ in board] == list(range(1, 51))
    scores = {s for _, s in board}
    assert len(scores) == 1  # exact tie, not just within rounding


N_GROUPS, GROUP_SIZE = 100, 100
USERS_PER_GROUP, RATINGS_PER_USER = 200, 10
N_ITEMS = N_GROUPS * GROUP_SIZE


def group_items(g):
    return range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)


def make_sparse_data(seed=7):
    """20k users / 10k items / 200k ratings; users rate only inside their
    own item group, so items from different groups never co-occur."""
    rng = random.Random(seed)
    ratings = []
    for g in range(N_GROUPS):
        items = list(group_items(g))
        for u in range(g * USERS_PER_GROUP, (g + 1) * USERS_PER_GROUP):
            for item in rng.sample(items, RATINGS_PER_USER):
                ratings.append((u, item, rng.randint(1, 5)))
    return ratings


def group_truth(ratings, g):
    lo = g * USERS_PER_GROUP
    truth = {}
    for u, i, r in ratings:
        if lo <= u < lo + USERS_PER_GROUP:
            truth.setdefault(u, {})[i] = r
    return truth


def test_sparse_scale_regression():
    import resource
    import sys

    ratings = make_sparse_data()
    index = ItemCFIndex()
    index.build(ratings)
    assert max(len(v) for v in index._ratings.values()) <= 30

    rng = random.Random(99)
    g0 = 0
    truth0 = group_truth(ratings, g0)
    users0 = sorted(truth0)
    touched_items = set()
    for _ in range(5):
        batch = []
        for u in rng.sample(users0, 10):
            rated = list(truth0[u])
            for item in rng.sample(rated, 3):
                r = rng.choice([None, 1, 2, 3, 4, 5])
                batch.append((u, item, r))
                touched_items.add(item)
        index.stats.clear()
        index.apply(batch)
        refreshes = index.stats["pair_refresh"]
        # locality: only pairs inside group 0 may be touched
        group0_pairs = sum(len(index._pairs[i]) for i in group_items(g0)) // 2
        total_pairs = sum(len(r) for r in index._pairs.values()) // 2
        assert 0 < refreshes <= group0_pairs
        assert refreshes < total_pairs // 10  # provably no global scan
        apply_to_truth(truth0, batch)

    # queries are O(returned rows): no pair/user work happens in neighbors()
    index.stats.clear()
    for g in range(N_GROUPS):
        index.neighbors(g * GROUP_SIZE)
    assert not index.stats, index.stats

    # correctness: group 0 matches a reference built from group-0 data only
    check_all(index, truth0, group_items(g0))
    # untouched groups still match their own reference
    for g in (1, 50, 99):
        check_all(index, group_truth(ratings, g), group_items(g))

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    assert peak_bytes < 1 << 30, f"peak RSS {peak_bytes / 2**30:.2f} GiB"


def test_apply_accepts_new_users_and_items_and_rebuild_resets():
    index = ItemCFIndex()
    index.build([(1, 1, 5), (1, 2, 1), (2, 1, 4), (2, 2, 2)])
    index.apply([(900, 800, 5), (900, 801, 4), (901, 800, 4), (901, 801, 5)])
    truth = {1: {1: 5, 2: 1}, 2: {1: 4, 2: 2},
             900: {800: 5, 801: 4}, 901: {800: 4, 801: 5}}
    check_all(index, truth, [1, 2, 800, 801, 802])
    # rebuild from scratch discards all previous state
    index.build([(5, 6, 3)])
    assert index.neighbors(1) == []
    assert index.neighbors(800) == []
    assert index.neighbors(6) == []
