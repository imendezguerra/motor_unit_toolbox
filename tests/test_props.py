import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from motor_unit_toolbox import props

# ---------------------------------------------------------------------------
# Firing properties
# ---------------------------------------------------------------------------


def test_check_mu_format_expands_1d():
    assert props._check_mu_format(np.zeros(10)).shape == (10, 1)
    assert props._check_mu_format(np.zeros((10, 3))).shape == (10, 3)


def _expected_dr(train, timestamps, silent_thr=0.25):
    """Reference implementation of the formula used by get_discharge_rate:
    n_spikes / (active period excluding ISIs > silent_thr)."""
    t = timestamps[train.astype(bool)]
    isi = np.diff(t)
    return len(t) / (t[-1] - t[0] - isi[isi > silent_thr].sum())


def test_discharge_rate_regular_trains(spike_train, timestamps):
    dr = props.get_discharge_rate(spike_train, timestamps)
    np.testing.assert_allclose(dr, [10, 20], rtol=0.1)
    expected = [_expected_dr(spike_train[:, i], timestamps) for i in range(2)]
    np.testing.assert_allclose(dr, expected)


def test_discharge_rate_empty_and_single_spike(timestamps):
    train = np.zeros((len(timestamps), 2), dtype=int)
    train[500, 1] = 1  # single spike -> zero active period
    np.testing.assert_array_equal(props.get_discharge_rate(train, timestamps), [0, 0])


def test_discharge_rate_excludes_silent_periods(make_train, timestamps):
    train = make_train(10)
    # Remove spikes in the middle to create a >0.25 s silent gap
    gap = slice(len(train) // 3, 2 * len(train) // 3)
    train_gap = train.copy()
    train_gap[gap] = 0
    dr = props.get_discharge_rate(train_gap, timestamps)
    # The gap is excluded from the active period, so the rate stays near 10 Hz
    # rather than dropping towards n_spikes / total_duration
    assert dr[0] == pytest.approx(_expected_dr(train_gap, timestamps))
    assert dr[0] > 8


@settings(max_examples=30, deadline=None)
@given(arrays(np.int8, st.tuples(st.integers(1, 200), st.integers(1, 5)), elements=st.integers(0, 1)))
def test_number_of_spikes_equals_sum(train):
    np.testing.assert_array_equal(props.get_number_of_spikes(train), train.sum(axis=0))


def test_inst_discharge_rate_shape(spike_train, fs):
    inst_dr = props.get_inst_discharge_rate(spike_train, fs=fs)
    assert inst_dr.shape == spike_train.shape
    assert np.all(inst_dr >= 0)
    # Away from the edges, a Hann-weighted 1 s window approximates the rate
    mid = slice(fs // 2 + 200, -fs // 2 - 200)
    np.testing.assert_allclose(inst_dr[mid].mean(axis=0), [10, 20], rtol=0.1)


def test_inst_discharge_rate_1d_input(make_train, fs):
    assert props.get_inst_discharge_rate(make_train(10), fs=fs).shape[-1] == 1


def test_cov_regular_trains_is_zero(spike_train, timestamps):
    cov = props.get_coefficient_of_variation(spike_train, timestamps)
    np.testing.assert_allclose(cov, 0, atol=1e-10)


def test_cov_empty_unit_is_nan(timestamps):
    train = np.zeros((len(timestamps), 1), dtype=int)
    assert np.isnan(props.get_coefficient_of_variation(train, timestamps)[0])


def test_cov_discard_isi(make_train, timestamps):
    train = make_train(10)
    train[len(train) // 3 : 2 * len(train) // 3] = 0  # long silent gap
    cov_discard = props.get_coefficient_of_variation(train, timestamps, discard_isi=0.25)
    cov_keep = props.get_coefficient_of_variation(train, timestamps, discard_isi=None)
    assert cov_discard[0] == pytest.approx(0, abs=1e-10)
    assert cov_keep[0] > 0.5


# ---------------------------------------------------------------------------
# IPT-based quality metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def ipts(spike_train, rng):
    clean = spike_train * 10.0 + 0.1 * np.abs(rng.standard_normal(spike_train.shape))
    noisy = spike_train * 10.0 + 4.0 * np.abs(rng.standard_normal(spike_train.shape))
    return clean, noisy


def test_pnr_higher_for_clean_ipt(spike_train, ipts):
    clean, noisy = ipts
    pnr_clean = props.get_pulse_to_noise_ratio(spike_train, clean)
    pnr_noisy = props.get_pulse_to_noise_ratio(spike_train, noisy)
    assert np.all(pnr_clean > pnr_noisy)
    assert np.all(pnr_clean > 30)


def test_pnr_no_spikes_is_nan(spike_train, ipts):
    empty = np.zeros_like(spike_train)
    assert np.all(np.isnan(props.get_pulse_to_noise_ratio(empty, ipts[0])))


def test_silhouette_clean_close_to_one(spike_train, ipts):
    clean, noisy = ipts
    sil_clean = props.get_silhouette_measure(spike_train, clean)
    sil_noisy = props.get_silhouette_measure(spike_train, noisy)
    assert np.all(sil_clean > 0.99)
    assert np.all(sil_clean >= sil_noisy)
    assert np.all((sil_clean <= 1) & (sil_noisy <= 1))


def test_silhouette_no_spikes_is_nan(spike_train, ipts):
    assert np.all(np.isnan(props.get_silhouette_measure(np.zeros_like(spike_train), ipts[0])))


def test_spike_baseline_amp(spike_train):
    ipts = spike_train * 5.0
    spikes_amp, base_amp = props.get_spike_baseline_amp(spike_train, ipts)
    np.testing.assert_allclose(spikes_amp, 5.0)
    np.testing.assert_allclose(base_amp, 0.0)


def test_find_reliable_units():
    dr = np.array([10, 2, 10, 10, 10, 50])
    cov = np.array([10, 10, 50, 10, 10, 10])
    sil = np.array([0.95, 0.95, 0.95, 0.5, 0.95, 0.95])
    pnr = np.array([35, 35, 35, 35, 20, 35])
    np.testing.assert_array_equal(
        props.find_reliable_units(dr, cov, sil, pnr),
        [True, False, False, False, False, False],
    )


# ---------------------------------------------------------------------------
# MUAP extraction and centring
# ---------------------------------------------------------------------------


def test_get_muaps_recovers_template(spike_train, emg, muap, fs):
    win_ms = 1000 * muap.shape[-1] / fs  # window matching the template length
    muaps = props.get_muaps(spike_train, emg, fs=fs, win_ms=win_ms)
    assert muaps.shape == (2, *muap.shape)
    np.testing.assert_allclose(muaps[0], muap, atol=1e-12)


def test_get_muaps_empty_spike_train(emg):
    muaps = props.get_muaps(np.empty(0), emg)
    assert muaps.shape[0] == 0


def test_center_muaps_moves_peak_to_centre(make_muap_fn):
    muap = make_muap_fn(center=10)
    centred = props.center_muaps(muap[None])
    assert np.argmax(np.abs(centred[0, 1, 2])) == muap.shape[-1] // 2


@pytest.mark.xfail(
    raises=(ValueError, AssertionError),
    reason="center_muaps unravels per-channel sample indices as channel indices "
    "(props.py center_muaps); fails when channels peak at different samples",
)
def test_center_muaps_channels_peaking_at_different_samples(make_muap_fn, pulse):
    muap = make_muap_fn(center=10)
    muap[0, 0] = pulse(40, 0.1)  # small background channel peaking late
    centred = props.center_muaps(muap[None])
    assert np.argmax(np.abs(centred[0, 1, 2])) == muap.shape[-1] // 2


def test_center_muaps_empty():
    empty = np.empty((0, 4, 4, 50))
    assert props.center_muaps(empty).shape == empty.shape


# ---------------------------------------------------------------------------
# MUAP features
# ---------------------------------------------------------------------------

# (function, expected ratio when amplitude is doubled)
AMPLITUDE_FEATURES = [
    (props.get_muap_ptp, 2.0),
    (props.get_muap_energy, 4.0),
    (props.get_muap_waveform_length, 2.0),
    (props.get_muap_ptp_time, 1.0),
]

FREQUENCY_FEATURES = [
    props.get_muap_peak_frequency,
    props.get_muap_median_frequency,
    props.get_muap_mean_frequency,
]


@pytest.fixture
def muap_pair(muap):
    """Two units sharing the dominant channel, the second at twice the amplitude."""
    return np.stack([muap, 2 * muap])


@pytest.mark.parametrize(("func", "ratio"), AMPLITUDE_FEATURES)
def test_amplitude_features_all_channels(func, ratio, muap_pair):
    out = func(muap_pair, sel_chs_by=None)
    assert out.shape == muap_pair.shape[:3]
    assert np.all(out >= 0)
    np.testing.assert_allclose(out[1], ratio * out[0])


@pytest.mark.parametrize(("func", "ratio"), AMPLITUDE_FEATURES)
@pytest.mark.parametrize("sel_chs_by", ["iqr", "iqr_ptp", "max_amp", "ptp"])
def test_amplitude_features_channel_selection(func, ratio, sel_chs_by, muap_pair):
    out = func(muap_pair, sel_chs_by=sel_chs_by)
    assert out.shape == muap_pair.shape[:3]
    # Only the dominant channel is selected; the rest are NaN
    assert np.all(~np.isnan(out[:, 1, 2]))
    assert np.isnan(out).sum() == out.size - 2
    assert out[1, 1, 2] == pytest.approx(ratio * out[0, 1, 2])


@pytest.mark.parametrize("func", [props.get_muap_ptp, props.get_muap_energy])
def test_amplitude_features_accept_single_muap(func, muap):
    assert func(muap, sel_chs_by=None).shape == (1, *muap.shape[:2])


@pytest.fixture
def tone_muaps(fs):
    """MUAPs with a pure cosine at an exact FFT bin on every channel."""
    samples = 64
    freq_hz = 8 * fs / samples  # bin 8 -> 256 Hz
    t = np.arange(samples) / fs
    wave = np.cos(2 * np.pi * freq_hz * t)
    muaps = np.tile(wave, (2, 4, 4, 1)) * 0.1
    muaps[:, 1, 2] *= 50  # dominant channel
    return muaps, freq_hz


@pytest.mark.parametrize("func", [props.get_muap_peak_frequency, props.get_muap_mean_frequency])
def test_frequency_features_pure_tone(func, tone_muaps, fs):
    muaps, freq_hz = tone_muaps
    out = func(muaps, sel_chs_by=None, fs=fs)
    assert out.shape == muaps.shape[:3]
    np.testing.assert_allclose(out, freq_hz, rtol=0.05)


@pytest.mark.parametrize("func", FREQUENCY_FEATURES)
def test_frequency_features_narrower_pulse_is_higher(func, pulse, fs):
    """A narrower pulse has more high-frequency content."""
    wide = np.tile(pulse(25, width=6.0), (1, 2, 2, 1))
    narrow = np.tile(pulse(25, width=1.0), (1, 2, 2, 1))
    f_wide = func(wide, sel_chs_by=None, fs=fs)
    f_narrow = func(narrow, sel_chs_by=None, fs=fs)
    assert np.all(f_narrow >= f_wide)


def test_mean_frequency_channel_selection(tone_muaps, fs):
    muaps, freq_hz = tone_muaps
    out = props.get_muap_mean_frequency(muaps, sel_chs_by="iqr", fs=fs)
    assert np.isnan(out).sum() == out.size - 2
    np.testing.assert_allclose(out[:, 1, 2], freq_hz, rtol=0.05)


@pytest.mark.xfail(
    raises=AssertionError,
    reason="`sel_chs_mask[row, col] is False` never matches a numpy bool, so "
    "non-selected channels are not masked (props.py peak/median frequency)",
)
@pytest.mark.parametrize("func", [props.get_muap_peak_frequency, props.get_muap_median_frequency])
def test_peak_median_frequency_channel_selection(func, tone_muaps, fs):
    muaps, _ = tone_muaps
    out = func(muaps, sel_chs_by="iqr", fs=fs)
    assert np.isnan(out).sum() == out.size - 2
