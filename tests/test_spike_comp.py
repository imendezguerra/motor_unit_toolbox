from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from motor_unit_toolbox import spike_comp as sc

FS = 2048
METRIC_COLS = ["roa", "f1", "precision", "sensitivity"]


def _train(seed: int, n: int = 100, isi: Tuple[int, int] = (60, 140), start: int = 200):
    """Irregular spike train: n spikes with uniform random inter-spike intervals (samples)."""
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.integers(*isi, n))


def _binary(n_samples: int, *trains: np.ndarray) -> np.ndarray:
    """Binary (n_samples, n_units) spike matrix of the given spike trains."""
    out = np.zeros((n_samples, len(trains)), dtype=np.int32)
    for unit, train in enumerate(trains):
        out[train, unit] = 1
    return out


def _binary_of(trains: Sequence[np.ndarray]) -> np.ndarray:
    """Binary spike matrix just long enough for the given trains."""
    return _binary(max(int(t.max()) for t in trains) + 1, *trains)


# ---------------------------------------------------------------------------
# Input formats
# ---------------------------------------------------------------------------


def test_spike_times_from_array_or_list():
    times = [np.array([300, 100]), np.array([], dtype=int)]
    expected = [[100, 300], []]
    assert [t.tolist() for t in sc._spike_times(times)] == expected
    assert [t.tolist() for t in sc._spike_times(_binary(400, *times))] == expected
    assert [t.tolist() for t in sc._spike_times(_binary(400, times[0])[:, 0])] == [[100, 300]]


def test_spike_times_list_must_be_integers():
    with pytest.raises(ValueError, match="integer"):
        sc._spike_times([np.array([0.1, 0.5])])


def test_list_and_array_inputs_agree():
    trains = [_train(0), _train(1) + 4]
    from_list = sc.compare_spikes([_train(1), _train(0)], trains, fs=FS)
    from_array = sc.compare_spikes(_binary_of([_train(1), _train(0)]), _binary_of(trains), fs=FS)
    pd.testing.assert_frame_equal(from_list, from_array)


@pytest.mark.parametrize(
    ("ref", "test", "match"),
    [
        (np.zeros((100, 2)), np.zeros((90, 2)), "Samples mismatch"),
        (np.zeros((100, 2)), np.zeros((100, 1)), "Units mismatch"),
    ],
)
def test_paired_mismatch_raises(ref, test, match):
    with pytest.raises(ValueError, match=match):
        sc.compare_paired_spikes(ref, test, fs=FS)


def test_unknown_metric_raises():
    trains = _binary_of([_train(0)])
    with pytest.raises(ValueError, match="metric"):
        sc.compute_agreement_matrix(trains, fs=FS, metric="accuracy")
    with pytest.raises(ValueError, match="metric"):
        sc.find_duplicates(trains, [0], fs=FS, metric="precision")


# ---------------------------------------------------------------------------
# Paired comparison and metrics
# ---------------------------------------------------------------------------


def test_metrics_count_matched_missed_and_extra_spikes():
    """Test train = reference + 5 samples, one reference spike missed, one extra spike."""
    ref = _binary(1000, np.array([100, 300, 520, 700, 910]))
    test = _binary(1000, np.array([105, 305, 525, 640, 915]))

    out = sc.compare_paired_spikes(ref, test, fs=FS, tol_spike_ms=1)

    assert out.iloc[0][["ref_idx", "test_idx", "lag", "tp", "fn", "fp"]].tolist() == [
        0,
        0,
        -5,
        4,
        1,
        1,
    ]
    np.testing.assert_allclose(out.loc[0, METRIC_COLS], [4 / 6, 0.8, 0.8, 0.8])


def test_lag_search_stays_within_max_lag():
    """A shift beyond max_lag is not aligned, between sets as well as paired."""
    t = _train(0)
    ref = _binary(t.max() + 300, t)
    near, far = _binary(ref.shape[0], t + 5), _binary(ref.shape[0], t + 120)  # 120 > 82 samples

    assert sc.compare_paired_spikes(ref, near, fs=FS)["lag"][0] == -5
    assert sc.compare_spikes(ref, near, fs=FS)["roa"][0] == 1.0
    paired_far = sc.compare_paired_spikes(ref, far, fs=FS)
    assert abs(paired_far["lag"][0]) <= round(40 / 1000 * FS)
    assert paired_far["roa"][0] < 0.3


@pytest.mark.parametrize(
    ("spikes", "kwargs", "expected"),
    [
        (_binary(1000, np.array([100, 500]), np.array([], int)), {}, [1.0, 0.0]),
        (_binary(50, np.array([5, 30])), {}, [1.0]),
        (_binary(1000, np.array([100, 500])), {"tol_spike_ms": 0.2}, [1.0]),
    ],
    ids=["unit-silent-in-both", "recording-shorter-than-max-lag", "tolerance-rounds-to-0"],
)
def test_paired_edge_cases_score_without_crashing(spikes, kwargs, expected):
    out = sc.compare_paired_spikes(spikes, spikes.copy(), fs=FS, **kwargs)
    np.testing.assert_allclose(out["roa"], expected)


spike_sets = st.sets(st.integers(min_value=0, max_value=1999), min_size=1, max_size=40)


@settings(max_examples=25, deadline=None)
@given(ref=spike_sets, test=spike_sets)
def test_scores_bounded_and_identity(ref, test):
    train_ref, train_test = (np.array(sorted(s)) for s in (ref, test))
    assert sc.compare_paired_spikes([train_ref], [train_ref], fs=FS)["roa"][0] == 1.0
    out = sc.compare_paired_spikes([train_ref], [train_test], fs=FS)
    assert out[METRIC_COLS].apply(lambda x: x.between(0, 1)).all(axis=None)
    assert out["tp"][0] + out["fn"][0] == len(ref)
    assert out["tp"][0] + out["fp"][0] == len(test)


# ---------------------------------------------------------------------------
# Unpaired comparison
# ---------------------------------------------------------------------------


def test_hungarian_pairs_exact_copy_over_busy_unit():
    """A unit holding all of a's spikes plus many extras must not take a's partner."""
    a, b = _train(0), _train(1)
    busy = np.union1d(a, _train(2, n=300, isi=(10, 30)))

    out = sc.compare_spikes([a, b], [b, busy, a], fs=FS)

    assert out[["ref_idx", "test_idx"]].values.tolist() == [[1, 0], [0, 2]]  # by test unit
    np.testing.assert_allclose(out["roa"], 1.0)


@pytest.mark.parametrize(
    ("min_score", "expected_pairs"), [(0.0, [[1, 0], [0, 1]]), (0.5, [[1, 0]])]
)
def test_compare_spikes_min_score_drops_poor_pairs(min_score, expected_pairs):
    """Reference 1 copies test 0; reference 0 shares a third of test 1's spikes."""
    a, b = _train(0), _train(1)
    part = np.sort(np.random.default_rng(1).choice(b, 30, replace=False))
    out = sc.compare_spikes([part, a], [a, b], fs=FS, min_score=min_score)
    assert out[["ref_idx", "test_idx"]].values.tolist() == expected_pairs


def test_compare_spikes_drops_pairs_without_common_spikes():
    a = np.arange(100, 5000, 100)
    out = sc.compare_spikes([a], [a + 50, a], fs=FS, max_lag_ms=1)
    assert out[["ref_idx", "test_idx"]].values.tolist() == [[0, 1]]
    assert sc.compare_spikes([a], [a + 50], fs=FS, max_lag_ms=1).empty


def test_compare_spikes_without_test_units_returns_no_pairs():
    out = sc.compare_spikes(np.zeros((50, 2), int), np.zeros((50, 0), int), fs=FS)
    assert out.empty
    assert list(out.columns) == ["ref_idx", "test_idx", "lag", "tp", "fn", "fp", *METRIC_COLS]


def test_compare_spikes_report_unmatched():
    a, b = _train(0), _train(1)
    out = sc.compare_spikes([a], [b, a + 3], fs=FS, report_unmatched=True)
    assert out["test_idx"].tolist() == [0, 1]
    assert out["ref_idx"].isna().tolist() == [True, False]
    assert out.loc[1, ["ref_idx", "lag", "tp", "fn", "fp"]].tolist() == [0, -3, len(a), 0, 0]
    # Unmatched test unit: all its spikes are false positives
    assert out.loc[0, ["tp", "fn", "fp"]].tolist() == [0, 0, len(b)]
    np.testing.assert_array_equal(out.loc[0, METRIC_COLS], 0)
    # Matched rows are those of the default output
    matched = out.dropna(subset="ref_idx").astype({"ref_idx": int, "lag": int})
    pd.testing.assert_frame_equal(
        matched.reset_index(drop=True), sc.compare_spikes([a], [b, a + 3], fs=FS)
    )


def test_compare_spikes_report_unmatched_without_ref_units():
    out = sc.compare_spikes(
        np.zeros((50, 0), int), np.eye(50, 2, dtype=int), fs=FS, report_unmatched=True
    )
    assert out["ref_idx"].isna().all()
    assert out["fp"].tolist() == [1, 1]


@pytest.mark.parametrize(("match_metric", "test_idx"), [("roa", 0), ("sensitivity", 1)])
def test_compare_spikes_match_metric(match_metric, test_idx):
    """RoA prefers 80 % of the reference spikes; sensitivity a busy unit holding all of them."""
    a = _train(0)
    busy = np.union1d(a, _train(2, n=300, isi=(10, 30)))
    out = sc.compare_spikes([a], [a[:80], busy], fs=FS, match_metric=match_metric)
    assert out["test_idx"].tolist() == [test_idx]


def test_agreement_matrix_between_and_within_sets():
    times = [_train(0), _train(0) + 3, _train(1), np.sort(_train(0)[::2] + 6)]
    score, lag = sc.compute_agreement_matrix(times, fs=FS)
    full_score, full_lag = sc.compute_agreement_matrix(times, times, fs=FS)
    np.testing.assert_allclose(score, full_score)
    np.testing.assert_allclose(score, score.T)
    np.testing.assert_array_equal(lag, -lag.T)
    np.testing.assert_array_equal(np.diag(score), 1)
    assert lag[0, 1] == full_lag[0, 1] == -3  # unit 1 = unit 0 + 3 samples
    assert sc.compute_agreement_matrix(times[:2], times, fs=FS)[0].shape == (2, 4)


# ---------------------------------------------------------------------------
# Duplicates within a set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quality", "kept_copy"), [([-0.2, -0.1, -0.3, -0.1], 1), ([0.95, 0.9, 0.97, 0.9], 2)]
)
def test_find_duplicates_collapses_every_copy_to_the_best_unit(quality, kept_copy):
    clean, other = _train(0), _train(1)
    rng = np.random.default_rng(3)
    copies = [clean, clean + 3, np.sort(rng.choice(clean, 70, replace=False)) + 6]

    kept_idx, info = sc.find_duplicates(_binary_of([*copies, other]), np.array(quality), fs=FS)

    np.testing.assert_array_equal(kept_idx, [kept_copy, 3])
    assert (info["kept_out_idx"] == 0).all()
    assert (info["kept_in_idx"] == kept_copy).all()
    assert sorted(info["duplicate_in_idx"]) == sorted({0, 1, 2} - {kept_copy})
    np.testing.assert_array_equal(info["quality_kept"], quality[kept_copy])
    np.testing.assert_array_equal(
        info["quality_duplicate"], np.array(quality)[info["duplicate_in_idx"]]
    )
    assert (info["roa"] >= 0.3).all()


def test_find_duplicates_info_lag_and_columns():
    a, b = _train(0), _train(1)
    kept_idx, info = sc.find_duplicates([b, a, a + 3], [0, 1, 0], fs=FS, metric="f1")
    np.testing.assert_array_equal(kept_idx, [0, 1])
    assert info.columns.tolist() == [
        "kept_out_idx",
        "kept_in_idx",
        "duplicate_in_idx",
        "lag",
        "f1",
        "quality_kept",
        "quality_duplicate",
    ]
    assert info.iloc[0][
        ["kept_out_idx", "kept_in_idx", "duplicate_in_idx", "lag", "f1"]
    ].tolist() == [1, 1, 2, -3, 1]


def test_find_duplicates_without_duplicates():
    kept_idx, info = sc.find_duplicates([_train(0), _train(1, start=237)], [0, 0], fs=FS)
    np.testing.assert_array_equal(kept_idx, [0, 1])
    assert info.empty


def test_find_duplicates_ranks_nan_quality_last_and_ties_by_unit_order():
    a = _train(0)
    np.testing.assert_array_equal(sc.find_duplicates([a, a], [np.nan, 0.5], fs=FS)[0], [1])
    np.testing.assert_array_equal(sc.find_duplicates([a, a], [0.5, 0.5], fs=FS)[0], [0])
    with pytest.raises(ValueError, match="quality"):
        sc.find_duplicates([a, a], [0.5], fs=FS)


def test_find_duplicates_bridge_unit_does_not_merge_two_units():
    """A unit mixing two motor units is a duplicate of both, which stay separate."""
    a, c = _train(0), _train(1, start=237)
    bridge = np.sort(np.concatenate([a[::2], c[1::2]]))
    kept_idx, _ = sc.find_duplicates([bridge, a, c], np.array([-0.5, -0.1, -0.1]), fs=FS)
    np.testing.assert_array_equal(kept_idx, [1, 2])


def test_find_duplicates_keeps_independent_fast_units():
    """Two unrelated ~30 Hz units: distinct at 0.5 ms, false duplicates at 10 ms."""
    a, b = _train(0, n=1800, isi=(55, 80)), _train(1, n=1800, isi=(55, 80))
    assert len(sc.find_duplicates([a, b], np.zeros(2), fs=FS)[0]) == 2
    assert len(sc.find_duplicates([a, b], np.zeros(2), fs=FS, tol_spike_ms=10)[0]) == 1


@pytest.mark.parametrize(("metric", "n_kept"), [("roa", 2), ("match", 2), ("f1", 1)])
def test_find_duplicates_metric(metric, n_kept):
    """A unit with 40 % of another's spikes: RoA = match = 0.4, F1 = 0.57."""
    a = _train(0)
    part = np.sort(np.random.default_rng(1).choice(a, 40, replace=False))
    kept_idx, _ = sc.find_duplicates([a, part], np.zeros(2), fs=FS, threshold=0.5, metric=metric)
    assert len(kept_idx) == n_kept


# ---------------------------------------------------------------------------
# Deprecated entry points
# ---------------------------------------------------------------------------


@pytest.fixture
def shifted_pair():
    t = _train(0)
    ref = _binary(t.max() + 300, t, _train(1))
    test = _binary(ref.shape[0], t + 5, _train(1))
    return ref, test


def test_deprecated_paired(shifted_pair):
    ref, test = shifted_pair
    expected = sc.compare_paired_spikes(ref, test, fs=2048, tol_spike_ms=1)
    with pytest.warns(FutureWarning, match="compare_paired_spikes"):
        roa, pair_idx, pair_lag = sc.rate_of_agreement_paired(ref, test)
    np.testing.assert_array_equal(roa, expected["roa"])
    assert pair_idx == [(0, 0), (1, 1)]
    np.testing.assert_array_equal(pair_lag, [-5, 0])
    assert pair_lag.dtype == float
    with pytest.warns(FutureWarning, match="compare_paired_spikes"):
        precision, sensitivity, f1, _, _ = sc.precision_sensitivity_f1_paired(ref, test)
    np.testing.assert_array_equal(
        [precision, sensitivity, f1], expected[["precision", "sensitivity", "f1"]].T
    )


def test_deprecated_unpaired(shifted_pair):
    ref, test = shifted_pair
    with pytest.warns(FutureWarning, match="compare_spikes"):
        roa, pair_idx, pair_lag = sc.rate_of_agreement(ref, test[:, ::-1])
    np.testing.assert_allclose(roa, 1.0)
    assert pair_idx == [(0, 1), (1, 0)]  # ordered by reference unit, as before
    assert pair_lag == [-5, 0]
    with pytest.warns(FutureWarning, match="compute_agreement_matrix"):
        roa, lag = sc.rate_of_agreement_full(ref, test)
    assert roa.shape == lag.shape == (2, 2)
    assert lag.dtype == float


def test_deprecated_within_set():
    a, b = _train(0), _train(1)
    trains = _binary_of([a, b, a + 10, b + 4])
    with pytest.warns(FutureWarning, match="find_duplicates"):
        roa, pair_idx, pair_lag = sc.rate_of_agreement(None, trains)
    assert pair_idx == [(0, 2), (1, 3)]
    np.testing.assert_allclose(roa, 1.0)
    assert pair_lag == [-10, -4]
    # The lag follows the (lower, higher) unit order whichever unit is kept
    with pytest.warns(FutureWarning):
        assert sc.rate_of_agreement(None, trains[:, [2, 0]])[1:] == ([(0, 1)], [10])
    for tol_train_ms in (40, None):
        with pytest.warns(FutureWarning, match="compute_agreement_matrix"):
            roa, lag = sc.rate_of_agreement_all(trains, tol_train_ms=tol_train_ms)
        assert roa[0, 2] == 1.0
        assert lag[0, 2] == -10


def test_deprecated_within_set_shapes():
    with pytest.warns(FutureWarning):
        roa, lag = sc.rate_of_agreement_all(np.zeros((100, 1), int))  # silent unit
    np.testing.assert_array_equal(roa, [1])
    np.testing.assert_array_equal(lag, [0])
    with pytest.warns(FutureWarning):
        assert sc.rate_of_agreement_all(np.empty(0))[0].shape == (0,)
