import networkx as nx
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from motor_unit_toolbox import muap_comp as mc

DIST_METRICS = ["nmse", "corr", "cosine", "nfd"]

# (ch_sel_method, ch_sel_metric, ch_sel_param): every method with both metrics
CH_SELS = [
    ("iqr", "abs", None),
    ("iqr", "ptp", None),
    ("perc", "abs", None),
    ("perc", "ptp", 0.9),
    ("n", "abs", 1),
    ("n", "ptp", 2),
    (None, "abs", None),
]

finite_floats = st.floats(-10, 10, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Channel selection and distance metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ch_sel", [c for c in CH_SELS if c[0] is not None and c[2] != 2])
def test_sel_chs_pick_dominant_channel(ch_sel, muap):
    mask = mc.get_sel_chs(muap, *ch_sel)
    assert mask.shape == muap.shape[:2]
    assert mask[1, 2]
    assert mask.sum() == 1


def test_sel_chs_all_and_n(muap):
    assert mc.get_sel_chs(muap, None).all()
    mask = mc.get_sel_chs(muap, "n", ch_sel_param=3)
    assert mask.sum() == 3
    assert mask[1, 2]
    assert mc.get_sel_chs(muap, "n").sum() == 5  # default n


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ch_sel_method": "max"}, "ch_sel_method"),
        ({"ch_sel_metric": "rms"}, "ch_sel_metric"),
        ({"ch_sel_method": "perc", "ch_sel_param": 95}, r"range \[0, 1\]"),
        ({"ch_sel_method": "iqr", "ch_sel_param": 90}, "ch_sel_param"),
    ],
)
def test_sel_chs_rejects_bad_options(kwargs, match, muap):
    with pytest.raises(ValueError, match=match):
        mc.get_sel_chs(muap, **kwargs)


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


def test_nfd_2d_uses_l2_norm_of_all_samples(rng):
    a, b = rng.standard_normal((2, 3, 20))
    a_norm = (a - a.mean(axis=-1, keepdims=True)) / np.sqrt(np.sum(a**2))
    b_norm = (b - b.mean(axis=-1, keepdims=True)) / np.sqrt(np.sum(b**2))
    expected = 2 * np.sum((a_norm - b_norm) ** 2) / (np.sum(a_norm**2) + np.sum(b_norm**2))
    assert mc.norm_farina_distance(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Alignment and MUAP distance/similarity
# ---------------------------------------------------------------------------


def test_alignment_recovers_shift(muap):
    shifted = np.roll(muap, -3, axis=-1)
    lag = mc.get_alignment(muap, shifted)
    assert isinstance(lag, int)
    np.testing.assert_allclose(np.roll(shifted, lag, axis=-1), muap, atol=1e-12)


def test_alignment_ignores_inverted_match(muap):
    # An inverted copy is anti-correlated at its true shift, which is not a match
    assert mc.get_alignment(muap, -np.roll(muap, -3, axis=-1)) != 3


@pytest.mark.parametrize("dist_metric", DIST_METRICS)
def test_muaps_dist_identity_and_shift(dist_metric, muap):
    assert mc.compute_muaps_dist(muap, muap, dist_metric) == (pytest.approx(0, abs=1e-10), 0)
    dist, lag = mc.compute_muaps_dist(muap, np.roll(muap, 4, axis=-1), dist_metric)
    assert dist == pytest.approx(0, abs=1e-10)
    assert lag == -4


@pytest.mark.parametrize("ch_sel", CH_SELS)
def test_muaps_dist_channel_selection(ch_sel, muaps):
    same, _ = mc.compute_muaps_dist(muaps[0], muaps[0], "nmse", *ch_sel)
    diff, _ = mc.compute_muaps_dist(muaps[0], muaps[1], "nmse", *ch_sel)
    assert same == pytest.approx(0, abs=1e-10)
    assert diff > 0.5


def test_muaps_dist_ignores_unselected_channels(muaps):
    # Only channel (0, 0) is selected: flipping a background channel changes nothing
    other = muaps[0].copy()
    other[3, 3] *= -1
    assert mc.compute_muaps_dist(muaps[0], other) == (pytest.approx(0, abs=1e-10), 0)
    assert mc.compute_muaps_dist(muaps[0], other, ch_sel_method=None)[0] > 0


@pytest.mark.parametrize(("arg", "value"), [("dist_metric", "euclidean"), ("ch_sel_method", "max")])
def test_muaps_dist_rejects_unknown_option(arg, value, muaps):
    with pytest.raises(ValueError, match=arg):
        mc.compute_muaps_dist(muaps[0], muaps[1], **{arg: value})


@pytest.mark.parametrize("dist_metric", DIST_METRICS)
def test_muaps_similarity_identity(dist_metric, muap):
    sim, lag = mc.compute_muaps_similarity(muap, muap, dist_metric)
    assert sim == pytest.approx(1)
    assert lag == 0


@pytest.mark.parametrize("ch_sel", CH_SELS)
@pytest.mark.parametrize("dist_metric", DIST_METRICS)
def test_muaps_similarity_is_one_minus_dist(dist_metric, ch_sel, muaps):
    sim, sim_lag = mc.compute_muaps_similarity(muaps[0], muaps[1], dist_metric, *ch_sel)
    dist, dist_lag = mc.compute_muaps_dist(muaps[0], muaps[1], dist_metric, *ch_sel)
    assert sim == pytest.approx(1 - dist)
    assert sim_lag == dist_lag


def test_muaps_dist_sets(muaps):
    dist, lags = mc.compute_muaps_dist_sets(muaps, muaps[:2])
    assert dist.shape == lags.shape == (3, 2)
    assert lags.dtype.kind == "i"
    np.testing.assert_allclose(np.diag(dist), 0, atol=1e-10)
    assert dist[2, 0] > 0.5
    assert mc.compute_muaps_dist_sets(muaps[0], muaps)[0].shape == (1, 3)  # single MUAP
    assert mc.compute_muaps_dist_sets(np.empty((0, 4, 4, 50)), muaps)[0].shape == (0, 0)


def test_all_muaps_dist(muaps, muap):
    dist, lags = mc.compute_all_muaps_dist(muaps)
    assert dist.shape == lags.shape == (3, 3)
    np.testing.assert_allclose(dist, dist.T)
    np.testing.assert_allclose(np.diag(dist), 0)
    assert np.all(dist[~np.eye(3, dtype=bool)] > 0.5)
    np.testing.assert_array_equal(mc.compute_all_muaps_dist(muap)[0], [[0]])  # single MUAP
    assert mc.compute_all_muaps_dist(np.empty((0, 4, 4, 50)))[0].shape == (0, 0)


def test_all_muaps_dist_lags_are_antisymmetric(muap):
    _, lags = mc.compute_all_muaps_dist(np.stack([muap, np.roll(muap, 3, axis=-1)]))
    np.testing.assert_array_equal(lags, [[0, -3], [3, 0]])


@pytest.mark.parametrize("ch_sel", CH_SELS)
@pytest.mark.parametrize("dist_metric", DIST_METRICS)
def test_dist_matrices_accept_every_option(dist_metric, ch_sel, muaps):
    shifted = np.roll(muaps, 2, axis=-1)
    dist, _ = mc.compute_all_muaps_dist(muaps, dist_metric, *ch_sel)
    sets_dist, sets_lags = mc.compute_muaps_dist_sets(muaps, shifted, dist_metric, *ch_sel)
    np.testing.assert_allclose(np.diag(dist), 0)
    np.testing.assert_allclose(np.diag(sets_dist), 0, atol=1e-10)
    np.testing.assert_array_equal(np.diag(sets_lags), -2)
    assert np.all(dist[~np.eye(3, dtype=bool)] > 0.1)


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


@pytest.mark.parametrize("assign", [mc.apply_hungarian_algorithm, mc.apply_sorted_grid_search])
def test_assignment_never_matches_nan(assign):
    # A flat MUAP gives NaN distances (e.g. corr): its row is left unmatched
    dist = np.array([[np.nan, np.nan], [0.1, 0.9]])
    rows, cols = assign(dist, dist_thr=0.3)
    assert (list(rows), list(cols)) == ([1], [0])


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
    assert sorted(zip(out.unit1, out.unit2, strict=True)) == [(0, 3), (1, 2), (2, 5), (3, 4)]
    assert sorted(map(sorted, mc.generate_group_sets(graph))) == [[0, 3, 4], [1, 2, 5]]
    assert dist_out[0, 3] == 2  # explored blocks are masked: ceil(max(1, 0.3)) + 1


@pytest.mark.parametrize(
    ("trial_labels", "trial_set"),
    [([1, 0, 0], [0, 1]), ([0, 1, 0], [0, 1]), ([0, 0, 2], [0, 1])],
    ids=["unsorted", "interleaved", "label-not-in-set"],
)
def test_assign_rejects_unsorted_trials(trial_labels, trial_set, muaps):
    with pytest.raises(ValueError, match="trial_set order"):
        mc.assign_muaps_across_seq_trials(np.ones((3, 3)), trial_labels, trial_set)
    with pytest.raises(ValueError, match="trial_set order"):
        mc.assign_muaps_all_trials(muaps, trial_labels, trial_set)


def test_assign_rejects_unknown_method(muaps):
    with pytest.raises(ValueError, match="assign_method"):
        mc.assign_muaps_all_trials(muaps, [0, 1, 2], [0, 1, 2], assign_method="greedy")


def test_assign_muaps_all_trials_single_unit(muap):
    labels, group_sets, df, _ = mc.assign_muaps_all_trials(muap[None], [0], [0])
    assert list(labels) == [1]
    assert group_sets == [[0]]
    assert len(df) == 0


@pytest.mark.slow
@pytest.mark.parametrize("assign_method", ["hungarian", "grid-search"])
@pytest.mark.parametrize("ch_sel", CH_SELS)
def test_assign_muaps_all_trials(assign_method, ch_sel, muaps):
    # Same 3 units in 3 trials, in a different order each time
    all_muaps = np.concatenate([muaps, muaps[[2, 0, 1]], muaps[[1, 2, 0]]])
    labels, group_sets, df, _ = mc.assign_muaps_all_trials(
        all_muaps, np.repeat([0, 1, 2], 3), [0, 1, 2], assign_method, "nmse", 0.2, *ch_sel
    )
    expected_groups = [{0, 4, 8}, {1, 5, 6}, {2, 3, 7}]
    assert sorted(map(set, group_sets), key=min) == expected_groups
    assert all(len({labels[i] for i in group}) == 1 for group in expected_groups)
    assert len(set(labels)) == 3
    assert len(df) == 6  # two links per tracked unit


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_mask_within_trial_dist():
    dist = np.array([
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.0, 0.4, 0.5],
        [0.2, 0.4, 0.0, 0.6],
        [0.3, 0.5, 0.6, 0.0],
    ])
    dist_before = dist.copy()

    masked = mc.mask_within_trial_dist(dist, np.array([0, 0, 1, 1]), fill_value=9)

    expected = dist.copy()
    expected[0, 1] = expected[1, 0] = expected[2, 3] = expected[3, 2] = 9
    np.testing.assert_array_equal(masked, expected)
    np.testing.assert_array_equal(dist, dist_before)  # input untouched
    # Default fill value: 1 + the largest distance, rounded up
    assert mc.mask_within_trial_dist(dist, [0, 0, 1, 1])[0, 1] == 2


def test_mask_within_trial_dist_rejects_wrong_length():
    with pytest.raises(ValueError, match="trial_labels"):
        mc.mask_within_trial_dist(np.zeros((3, 3)), np.array([0, 1]))


def test_cluster_muaps_rejects_no_valid_threshold():
    # Two units: no threshold gives between 2 and n_units - 1 clusters
    with pytest.raises(ValueError, match="thr_vals"):
        mc.cluster_muaps(np.array([[0.0, 0.5], [0.5, 0.0]]), flag_plot=False)


@pytest.mark.slow
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


@pytest.fixture
def same_trial_duplicates(muaps):
    """Two families of identical-shape MUAPs, each with a duplicate inside trial 0 or 1."""
    family = np.stack([muaps[0], 1.1 * muaps[0], 0.9 * muaps[0], muaps[1], 1.2 * muaps[1], 0.8 * muaps[1]])
    dist, _ = mc.compute_all_muaps_dist(family, dist_metric="corr")
    trial_labels = np.array([0, 0, 1, 0, 1, 1])  # units 0-1 and 4-5 share a trial
    return dist, trial_labels


@pytest.mark.slow
def test_cluster_muaps_complete_linkage_keeps_trials_apart(same_trial_duplicates):
    dist, trial_labels = same_trial_duplicates
    dist_before = dist.copy()

    # Any warning fails the test (filterwarnings = error), so this also checks none is raised
    opt, out = mc.cluster_muaps(
        dist, cluster_method="complete", thr_vals=np.arange(0.05, 2.0, 0.05),
        flag_plot=False, trial_labels=trial_labels,
    )

    labels = out["labels"][opt["opt_idx"].iloc[0]]
    for cluster in np.unique(labels):
        cluster_trials = trial_labels[labels == cluster]
        assert len(cluster_trials) == len(set(cluster_trials))
    np.testing.assert_array_equal(dist, dist_before)


@pytest.mark.slow
def test_cluster_muaps_warns_when_same_trial_units_merge(same_trial_duplicates):
    dist, trial_labels = same_trial_duplicates

    # Single linkage merges through a unit of another trial despite the masking
    with pytest.warns(UserWarning, match=r"units from the same trial") as record:
        mc.cluster_muaps(
            dist, cluster_method="single", thr_vals=np.arange(0.05, 2.0, 0.05),
            flag_plot=False, trial_labels=trial_labels,
        )

    message = str(record[0].message)
    assert "At the selected threshold" in message
    assert "units [0, 1] (trial 0)" in message
    assert "units [4, 5] (trial 1)" in message
    assert "cluster " in message
