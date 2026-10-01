import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from motor_unit_toolbox import spike_comp as sc


@pytest.fixture
def half_train(spike_train):
    """Copy of spike_train with every other spike removed."""
    half = spike_train.copy()
    for unit in range(half.shape[1]):
        half[np.nonzero(half[:, unit])[0][1::2], unit] = 0
    return half


def _pairs(pair_idx):
    return [tuple(int(i) for i in p) for p in pair_idx]


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------


def test_roa_paired_identical(spike_train, make_train):
    roa, pair_idx, pair_lag = sc.rate_of_agreement_paired(spike_train, spike_train)
    np.testing.assert_allclose(roa, 1.0)
    assert list(pair_idx) == [(0, 0), (1, 1)]
    np.testing.assert_array_equal(pair_lag, 0)
    # 1-D input is treated as a single unit
    np.testing.assert_allclose(sc.rate_of_agreement_paired(make_train(10), make_train(10))[0], [1.0])


def test_roa_paired_shift_within_tolerance(spike_train):
    shift = 20  # samples, inside the 40 ms train tolerance
    shifted = np.zeros_like(spike_train)
    shifted[shift:] = spike_train[:-shift]  # zero-pad (no wrap-around)
    roa, _, pair_lag = sc.rate_of_agreement_paired(spike_train, shifted)
    # Spikes pushed past the end are lost; every remaining one matches
    np.testing.assert_allclose(roa, shifted.sum(axis=0) / spike_train.sum(axis=0))
    np.testing.assert_array_equal(pair_lag, -shift)


def test_roa_paired_half_spikes(spike_train, half_train):
    roa, _, _ = sc.rate_of_agreement_paired(spike_train, half_train)
    np.testing.assert_allclose(roa, 0.5, atol=0.03)


def test_precision_sensitivity_f1(spike_train, half_train):
    precision, sensitivity, f1, _, _ = sc.precision_sensitivity_f1_paired(spike_train, spike_train)
    np.testing.assert_allclose([precision, sensitivity, f1], 1.0)

    # Half the reference spikes detected: no false positives, half missed
    precision, sensitivity, f1, _, _ = sc.precision_sensitivity_f1_paired(spike_train, half_train)
    np.testing.assert_allclose(precision, 1.0)
    np.testing.assert_allclose(sensitivity, 0.5, atol=0.03)
    np.testing.assert_allclose(f1, 2 * precision * sensitivity / (precision + sensitivity))


def test_tp_fp_fn(spike_train, half_train):
    tp, fp, fn, _, _ = sc.get_tp_fp_fn_paired(spike_train, half_train)
    assert tp.shape == fp.shape == fn.shape == spike_train.shape
    np.testing.assert_array_equal(tp.sum(axis=0), half_train.sum(axis=0))
    np.testing.assert_array_equal(fp.sum(axis=0), 0)
    np.testing.assert_array_equal(tp.sum(axis=0) + fn.sum(axis=0), spike_train.sum(axis=0))


# (function, number of leading outputs that are per-unit metrics)
METRIC_FUNCS = [
    (sc.rate_of_agreement_paired, 1),
    (sc.precision_sensitivity_f1_paired, 3),
    (sc.get_tp_fp_fn_paired, 3),
    (sc.rate_of_agreement, 1),
]


@pytest.mark.parametrize(("func", "n_metrics"), METRIC_FUNCS)
def test_no_test_spikes_gives_zeros(func, n_metrics, spike_train):
    out = func(spike_train, np.zeros_like(spike_train))
    for metric in out[:n_metrics]:
        np.testing.assert_array_equal(metric, 0)


@pytest.mark.parametrize(
    "func",
    [sc.rate_of_agreement_paired, sc.precision_sensitivity_f1_paired, sc.get_tp_fp_fn_paired],
)
def test_paired_shape_mismatch_raises(func, spike_train):
    with pytest.raises(ValueError, match="Dimensionality mismatch"):
        func(spike_train, spike_train[:, :1])


# ---------------------------------------------------------------------------
# Unpaired and within-set comparisons
# ---------------------------------------------------------------------------


def test_roa_unpaired_recovers_permutation(spike_train):
    roa, pair_idx, pair_lag = sc.rate_of_agreement(spike_train, spike_train[:, ::-1])
    np.testing.assert_allclose(roa, 1.0)
    assert _pairs(pair_idx) == [(0, 1), (1, 0)]
    assert pair_lag == [0, 0]


def test_roa_within_set_finds_duplicate(spike_train):
    # Unit 2 is unit 0 delayed by 10 samples; unit 1 is unrelated
    delayed = np.zeros_like(spike_train[:, 0])
    delayed[10:] = spike_train[:-10, 0]
    roa, pair_idx, pair_lag = sc.rate_of_agreement(None, np.column_stack([spike_train, delayed]))
    assert _pairs(pair_idx) == [(0, 2)]
    np.testing.assert_allclose(roa, 1.0)
    assert pair_lag == [-10]


def test_roa_full(spike_train, make_train):
    roa, lags = sc.rate_of_agreement_full(spike_train, np.column_stack([spike_train, make_train(15)]))
    assert roa.shape == lags.shape == (2, 3)
    np.testing.assert_allclose(np.diag(roa[:, :2]), 1.0)
    assert np.all((roa >= 0) & (roa <= 1))


@pytest.mark.parametrize("func", [sc.rate_of_agreement, sc.rate_of_agreement_full])
def test_unpaired_time_mismatch_raises(func, spike_train):
    with pytest.raises(ValueError, match="Time dimensionality mismatch"):
        func(spike_train, spike_train[:-1])


def test_roa_all(spike_train):
    trains = np.column_stack([spike_train, spike_train[:, 0]])  # unit 2 duplicates unit 0
    roa, lags = sc.rate_of_agreement_all(trains)
    assert roa.shape == lags.shape == (3, 3)
    np.testing.assert_allclose(np.diag(roa), 1.0)
    np.testing.assert_allclose(roa, roa.T)
    np.testing.assert_array_equal(lags, -lags.T)
    assert roa[0, 2] == pytest.approx(1.0)
    assert roa[0, 1] < 1.0


def test_roa_all_edge_cases(spike_train, make_train):
    roa, lags = sc.rate_of_agreement_all(make_train(10))
    np.testing.assert_array_equal(roa, [1])
    np.testing.assert_array_equal(lags, [0])
    assert sc.rate_of_agreement_all(np.empty(0))[0].size == 0
    with pytest.raises(Warning, match="Consider transposing"):
        sc.rate_of_agreement_all(spike_train.T)


# ---------------------------------------------------------------------------
# Property-based
# ---------------------------------------------------------------------------

spike_sets = st.sets(st.integers(min_value=0, max_value=1999), min_size=1, max_size=40)


@settings(max_examples=25, deadline=None)
@given(ref=spike_sets, test=spike_sets)
def test_roa_bounds_and_identity(ref, test):
    train_ref = np.zeros(2000, dtype=int)
    train_ref[list(ref)] = 1
    train_test = np.zeros(2000, dtype=int)
    train_test[list(test)] = 1

    np.testing.assert_allclose(sc.rate_of_agreement_paired(train_ref, train_ref)[0], 1.0)
    roa = sc.rate_of_agreement_paired(train_ref, train_test)[0]
    assert np.all((roa >= 0) & (roa <= 1))
