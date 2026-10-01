import networkx as nx
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from motor_unit_toolbox import muap_comp as mc

CH_SELECTORS = [
    mc.get_highest_iqr_ch,
    mc.get_highest_iqr_ptp_ch,
    mc.get_highest_amp_ch,
    mc.get_highest_ptp_ch,
    mc.get_percentile_ch,
]

DIST_METRICS = ["nmse", "corr", "cosine", "nfd"]
SEL_CHS = ["iqr", "iqr_ptp", "max_abs", "ptp"]

finite_floats = st.floats(-10, 10, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selector", CH_SELECTORS)
def test_channel_selectors_pick_dominant_channel(selector, muap):
    mask = selector(muap)
    assert mask.shape == muap.shape[:2]
    assert mask.dtype == bool
    assert mask[1, 2]
    assert mask.sum() == 1


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


def test_nmse_identity(muap):
    assert mc.nmse(muap, muap) == pytest.approx(0)


@settings(max_examples=30, deadline=None)
@given(
    a=arrays(np.float64, (3, 8), elements=finite_floats),
    b=arrays(np.float64, (3, 8), elements=finite_floats),
)
def test_nmse_symmetric_and_non_negative(a, b):
    # nmse is 0/0 when both signals have (numerically) zero energy, e.g. all
    # zeros or values so small that squaring them underflows
    assume(np.sum(a**2) + np.sum(b**2) > 1e-12)
    assert mc.nmse(a, b) == pytest.approx(mc.nmse(b, a))
    assert mc.nmse(a, b) >= 0


def test_nfd_identity(rng):
    a = rng.standard_normal((3, 20))
    assert mc.norm_farina_distance(a, a) == pytest.approx(0)


def test_nfd_accepts_1d(rng):
    a = rng.standard_normal(20)
    assert mc.norm_farina_distance(a, a) == pytest.approx(0)


def test_nfd_symmetric(rng):
    a = rng.standard_normal((3, 20))
    b = 3 * rng.standard_normal((3, 20))
    assert mc.norm_farina_distance(a, b) == pytest.approx(mc.norm_farina_distance(b, a))


# ---------------------------------------------------------------------------
# Alignment and MUAP distance/similarity
# ---------------------------------------------------------------------------


def test_alignment_recovers_shift(muap):
    shifted = np.roll(muap, -3, axis=-1)
    lag = mc.get_alignmnent(muap, shifted)
    np.testing.assert_allclose(np.roll(shifted, lag, axis=-1), muap, atol=1e-12)


def test_alignment_identity(muap):
    assert mc.get_alignmnent(muap, muap) == 0


@pytest.mark.parametrize("metric", DIST_METRICS)
@pytest.mark.parametrize("sel_chs_by", SEL_CHS)
def test_muaps_dist_identity(metric, sel_chs_by, muap):
    dist, lag = mc.compute_muaps_dist(muap, muap, sel_chs_by=sel_chs_by, metric=metric)
    assert dist == pytest.approx(0, abs=1e-10)
    assert lag == 0


@pytest.mark.parametrize("metric", DIST_METRICS)
def test_muaps_dist_shifted_copy(metric, muap):
    dist, lag = mc.compute_muaps_dist(muap, np.roll(muap, 4, axis=-1), metric=metric)
    assert dist == pytest.approx(0, abs=1e-10)
    assert lag == -4


def test_muaps_dist_different_units(muaps):
    same, _ = mc.compute_muaps_dist(muaps[0], muaps[0])
    diff, _ = mc.compute_muaps_dist(muaps[0], muaps[1])
    assert diff > same + 0.5


@pytest.mark.parametrize("metric", DIST_METRICS)
def test_muaps_similarity_identity(metric, muap):
    sim, lag = mc.compute_muaps_similarity(muap, muap, metric=metric)
    assert sim == pytest.approx(1)
    assert lag == 0


def test_muaps_similarity_default_metric_is_nmse(muap):
    sim, lag = mc.compute_muaps_similarity(muap, muap)
    assert sim == pytest.approx(1)
    assert lag == 0


def test_muaps_similarity_nmse_matches_distance(muaps):
    sim, sim_lag = mc.compute_muaps_similarity(muaps[0], muaps[1], metric="nmse")
    dist, dist_lag = mc.compute_muaps_dist(muaps[0], muaps[1], metric="nmse")
    assert sim == pytest.approx(1 - dist)
    assert sim_lag == dist_lag


def test_muaps_dist_sets(muaps):
    dist, lags = mc.compute_muaps_dist_sets(muaps, muaps[:2])
    assert dist.shape == lags.shape == (3, 2)
    assert lags.dtype.kind == "i"
    np.testing.assert_allclose(np.diag(dist), 0, atol=1e-10)
    assert dist[2, 0] > 0.5


def test_muaps_dist_sets_single_muap(muaps):
    dist, _ = mc.compute_muaps_dist_sets(muaps[0], muaps)
    assert dist.shape == (1, 3)


def test_muaps_dist_sets_empty(muaps):
    dist, lags = mc.compute_muaps_dist_sets(np.empty((0, 4, 4, 50)), muaps)
    assert dist.size == 0
    assert lags.size == 0


def test_all_muaps_dist(muaps):
    dist, lags = mc.compute_all_muaps_dist(muaps)
    assert dist.shape == lags.shape == (3, 3)
    np.testing.assert_allclose(dist, dist.T)
    np.testing.assert_allclose(np.diag(dist), 0)
    assert np.all(dist[~np.eye(3, dtype=bool)] > 0.5)


def test_all_muaps_dist_edge_cases(muap):
    dist, lags = mc.compute_all_muaps_dist(muap)
    np.testing.assert_array_equal(dist, [0])
    dist, _ = mc.compute_all_muaps_dist(np.empty((0, 4, 4, 50)))
    assert dist.size == 0


# ---------------------------------------------------------------------------
# Assignment helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def perm_dist():
    """3x3 distance matrix whose optimal assignment is the permutation [2, 0, 1]."""
    dist = np.full((3, 3), 0.9)
    dist[0, 2] = 0.1
    dist[1, 0] = 0.2
    dist[2, 1] = 0.5
    return dist


@pytest.mark.parametrize("assign", [mc.apply_hungarian_algorithm, mc.apply_sorted_grid_search])
def test_assignment_recovers_permutation(assign, perm_dist):
    rows, cols = assign(perm_dist, dist_thr=1.0)
    assert list(rows) == [0, 1, 2]
    assert list(cols) == [2, 0, 1]


@pytest.mark.parametrize("assign", [mc.apply_hungarian_algorithm, mc.apply_sorted_grid_search])
def test_assignment_respects_threshold(assign, perm_dist):
    rows, cols = assign(perm_dist, dist_thr=0.3)
    assert list(rows) == [0, 1]
    assert list(cols) == [2, 0]


def test_pairwise():
    assert list(mc.pairwise("ABCD")) == [("A", "B"), ("B", "C"), ("C", "D")]
    assert list(mc.pairwise([1])) == []


def test_generate_group_sets():
    graph = nx.DiGraph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 3), (3, 4), (1, 5)])
    sets = mc.generate_group_sets(graph)
    assert sets[0] == [0, 3, 4]
    assert sets[1] == [1, 5]
    assert sorted(map(tuple, sets[2:])) == [(2,)]


def test_generate_group_labels():
    labels = mc.generate_group_labels([[0, 3, 4], [1, 5], [2]])
    np.testing.assert_array_equal(labels, [1, 2, 3, 1, 1, 2])


@pytest.mark.xfail(
    raises=IndexError,
    reason="generate_group_labels docstring example uses 1-based ids; "
    "the function assumes contiguous 0-based unit ids",
)
def test_generate_group_labels_docstring_example():
    np.testing.assert_array_equal(mc.generate_group_labels([[1, 2, 3, 4]]), [1, 1, 1, 1])


def test_assign_muaps_across_seq_trials():
    # 3 trials x 2 units. Matches: 0->3->4 and 1->2->5
    trial_labels = np.array([0, 0, 1, 1, 2, 2])
    dist = np.ones((6, 6))
    for i, j in [(0, 3), (1, 2), (3, 4), (2, 5)]:
        dist[i, j] = 0.05

    out, graph, dist_out = mc.assign_muaps_across_seq_trials(
        dist.copy(), trial_labels, trial_set=[0, 1, 2], dist_thr=0.3
    )
    assert sorted(zip(out.unit1, out.unit2)) == [(0, 3), (1, 2), (2, 5), (3, 4)]
    assert all(d == pytest.approx(0.05) for d in out.link_dist)
    assert sorted(map(sorted, mc.generate_group_sets(graph))) == [[0, 3, 4], [1, 2, 5]]
    # Explored blocks are masked
    assert dist_out[0, 3] == 2


@pytest.mark.slow
@pytest.mark.parametrize("assign_method", ["hungarian", "grid-search"])
def test_assign_muaps_all_trials_tracks_units(assign_method, muaps):
    # Same 3 units in 3 trials, in a different order each time
    all_muaps = np.concatenate([muaps, muaps[[2, 0, 1]], muaps[[1, 2, 0]]])
    trial_labels = np.repeat([0, 1, 2], 3)

    labels, group_sets, df, _ = mc.assign_muaps_all_trials(
        all_muaps, trial_labels, trial_set=[0, 1, 2], assign_method=assign_method
    )
    expected_groups = [{0, 4, 8}, {1, 5, 6}, {2, 3, 7}]
    assert sorted(map(set, group_sets), key=min) == expected_groups
    for group in expected_groups:
        assert len({labels[i] for i in group}) == 1
    assert len(set(labels)) == 3
    assert len(df) == 6  # two links per tracked unit


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@pytest.mark.slow
# At thresholds where every unit falls into one cluster the between-cluster
# block is empty, so nanmean/nanstd warn and return NaN (expected)
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
def test_cluster_muaps_finds_two_clusters(muaps):
    family = np.stack([muaps[0], 1.1 * muaps[0], 0.9 * muaps[0], muaps[1], 1.2 * muaps[1], 0.8 * muaps[1]])
    dist, _ = mc.compute_all_muaps_dist(family, dist_metric="corr")
    dist_before = dist.copy()

    thr_vals = np.arange(0.05, 2.0, 0.05)
    opt, out = mc.cluster_muaps(dist, thr_vals=thr_vals, flag_plot=False)

    assert opt["opt_n_clusters"].iloc[0] == 2
    labels = out["labels"][opt["opt_idx"].iloc[0]]
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert out["labels"].shape == (len(thr_vals), 6)
    # The distance matrix diagonal is restored after use
    np.testing.assert_array_equal(dist, dist_before)
