import networkx as nx
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from motor_unit_toolbox import muap_comp as mc

DIST_METRICS = ["nmse", "corr", "cosine", "nfd"]

finite_floats = st.floats(-10, 10, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Channel selection and distance metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selector",
    [
        mc.get_highest_iqr_ch,
        mc.get_highest_iqr_ptp_ch,
        mc.get_highest_amp_ch,
        mc.get_highest_ptp_ch,
        mc.get_percentile_ch,
    ],
)
def test_channel_selectors_pick_dominant_channel(selector, muap):
    mask = selector(muap)
    assert mask.shape == muap.shape[:2]
    assert mask[1, 2]
    assert mask.sum() == 1


@settings(max_examples=30, deadline=None)
@given(
    a=arrays(np.float64, (3, 8), elements=finite_floats),
    b=arrays(np.float64, (3, 8), elements=finite_floats),
)
def test_nmse_symmetric_and_non_negative(a, b):
    # 0/0 when both signals have (numerically) zero energy
    assume(np.sum(a**2) + np.sum(b**2) > 1e-12)
    assert mc.nmse(a, b) == pytest.approx(mc.nmse(b, a))
    assert mc.nmse(a, b) >= 0


def test_nfd(rng):
    a = rng.standard_normal((3, 20))
    b = 3 * rng.standard_normal((3, 20))
    assert mc.norm_farina_distance(a, a) == pytest.approx(0)
    assert mc.norm_farina_distance(a[0], a[0]) == pytest.approx(0)  # 1-D input
    assert mc.norm_farina_distance(a, b) == pytest.approx(mc.norm_farina_distance(b, a))


# ---------------------------------------------------------------------------
# Alignment and MUAP distance/similarity
# ---------------------------------------------------------------------------


def test_alignment_recovers_shift(muap):
    shifted = np.roll(muap, -3, axis=-1)
    lag = mc.get_alignmnent(muap, shifted)
    np.testing.assert_allclose(np.roll(shifted, lag, axis=-1), muap, atol=1e-12)


@pytest.mark.parametrize("metric", DIST_METRICS)
def test_muaps_dist_identity_and_shift(metric, muap):
    assert mc.compute_muaps_dist(muap, muap, metric=metric) == (pytest.approx(0, abs=1e-10), 0)
    dist, lag = mc.compute_muaps_dist(muap, np.roll(muap, 4, axis=-1), metric=metric)
    assert dist == pytest.approx(0, abs=1e-10)
    assert lag == -4


@pytest.mark.parametrize("sel_chs_by", ["iqr", "iqr_ptp", "max_abs", "ptp"])
def test_muaps_dist_channel_selection(sel_chs_by, muaps):
    same, _ = mc.compute_muaps_dist(muaps[0], muaps[0], sel_chs_by=sel_chs_by)
    diff, _ = mc.compute_muaps_dist(muaps[0], muaps[1], sel_chs_by=sel_chs_by)
    assert same == pytest.approx(0, abs=1e-10)
    assert diff > 0.5


@pytest.mark.parametrize("metric", DIST_METRICS)
def test_muaps_similarity_identity(metric, muap):
    sim, lag = mc.compute_muaps_similarity(muap, muap, metric=metric)
    assert sim == pytest.approx(1)
    assert lag == 0


def test_muaps_similarity_default_is_one_minus_nmse(muaps):
    sim, sim_lag = mc.compute_muaps_similarity(muaps[0], muaps[1])
    dist, dist_lag = mc.compute_muaps_dist(muaps[0], muaps[1], metric="nmse")
    assert sim == pytest.approx(1 - dist)
    assert sim_lag == dist_lag


def test_muaps_dist_sets(muaps):
    dist, lags = mc.compute_muaps_dist_sets(muaps, muaps[:2])
    assert dist.shape == lags.shape == (3, 2)
    assert lags.dtype.kind == "i"
    np.testing.assert_allclose(np.diag(dist), 0, atol=1e-10)
    assert dist[2, 0] > 0.5
    assert mc.compute_muaps_dist_sets(muaps[0], muaps)[0].shape == (1, 3)  # single MUAP
    assert mc.compute_muaps_dist_sets(np.empty((0, 4, 4, 50)), muaps)[0].size == 0


def test_all_muaps_dist(muaps, muap):
    dist, lags = mc.compute_all_muaps_dist(muaps)
    assert dist.shape == lags.shape == (3, 3)
    np.testing.assert_allclose(dist, dist.T)
    np.testing.assert_allclose(np.diag(dist), 0)
    assert np.all(dist[~np.eye(3, dtype=bool)] > 0.5)
    np.testing.assert_array_equal(mc.compute_all_muaps_dist(muap)[0], [0])
    assert mc.compute_all_muaps_dist(np.empty((0, 4, 4, 50)))[0].size == 0


# ---------------------------------------------------------------------------
# Assignment and tracking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("assign", [mc.apply_hungarian_algorithm, mc.apply_sorted_grid_search])
def test_assignment(assign):
    # Optimal assignment is the permutation [2, 0, 1]
    dist = np.full((3, 3), 0.9)
    dist[0, 2], dist[1, 0], dist[2, 1] = 0.1, 0.2, 0.5

    rows, cols = assign(dist, dist_thr=1.0)
    assert (list(rows), list(cols)) == ([0, 1, 2], [2, 0, 1])
    rows, cols = assign(dist, dist_thr=0.3)  # drops the 0.5 match
    assert (list(rows), list(cols)) == ([0, 1], [2, 0])


def test_group_sets_and_labels():
    graph = nx.DiGraph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 3), (3, 4), (1, 5)])
    sets = mc.generate_group_sets(graph)
    assert sets == [[0, 3, 4], [1, 5], [2]]  # sorted by size
    np.testing.assert_array_equal(mc.generate_group_labels(sets), [1, 2, 3, 1, 1, 2])
    # Docstring example
    np.testing.assert_array_equal(mc.generate_group_labels([[0, 2, 3], [1]]), [1, 2, 1, 1])


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
    assert sorted(map(sorted, mc.generate_group_sets(graph))) == [[0, 3, 4], [1, 2, 5]]
    assert dist_out[0, 3] == 2  # explored blocks are masked


@pytest.mark.slow
@pytest.mark.parametrize("assign_method", ["hungarian", "grid-search"])
def test_assign_muaps_all_trials(assign_method, muaps):
    # Same 3 units in 3 trials, in a different order each time
    all_muaps = np.concatenate([muaps, muaps[[2, 0, 1]], muaps[[1, 2, 0]]])
    labels, group_sets, df, _ = mc.assign_muaps_all_trials(
        all_muaps, np.repeat([0, 1, 2], 3), trial_set=[0, 1, 2], assign_method=assign_method
    )
    expected_groups = [{0, 4, 8}, {1, 5, 6}, {2, 3, 7}]
    assert sorted(map(set, group_sets), key=min) == expected_groups
    assert all(len({labels[i] for i in group}) == 1 for group in expected_groups)
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
def test_cluster_muaps(muaps):
    family = np.stack([muaps[0], 1.1 * muaps[0], 0.9 * muaps[0], muaps[1], 1.2 * muaps[1], 0.8 * muaps[1]])
    dist, _ = mc.compute_all_muaps_dist(family, dist_metric="corr")
    dist_before = dist.copy()

    opt, out = mc.cluster_muaps(dist, thr_vals=np.arange(0.05, 2.0, 0.05), flag_plot=False)

    assert opt["opt_n_clusters"].iloc[0] == 2
    labels = out["labels"][opt["opt_idx"].iloc[0]]
    assert len(set(labels[:3])) == len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    np.testing.assert_array_equal(dist, dist_before)  # diagonal restored after use
