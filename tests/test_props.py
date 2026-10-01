import numpy as np
import pytest

from motor_unit_toolbox import props

# ---------------------------------------------------------------------------
# Firing properties
# ---------------------------------------------------------------------------


def _expected_dr(train, timestamps, silent_thr=0.25):
    """Formula used by get_discharge_rate:
    n_spikes / (active period excluding ISIs > silent_thr)."""
    t = timestamps[train.astype(bool)]
    isi = np.diff(t)
    return len(t) / (t[-1] - t[0] - isi[isi > silent_thr].sum())


def test_discharge_rate(spike_train, make_train, timestamps):
    dr = props.get_discharge_rate(spike_train, timestamps)
    np.testing.assert_allclose(dr, [10, 20], rtol=0.1)
    np.testing.assert_allclose(dr, [_expected_dr(spike_train[:, i], timestamps) for i in range(2)])

    # A >0.25 s silent gap is excluded from the active period
    gap = make_train(10)
    gap[len(gap) // 3 : 2 * len(gap) // 3] = 0
    assert props.get_discharge_rate(gap, timestamps)[0] == pytest.approx(
        _expected_dr(gap, timestamps)
    )


def test_discharge_rate_no_or_single_spike(timestamps):
    train = np.zeros((len(timestamps), 2), dtype=int)
    train[500, 1] = 1
    np.testing.assert_array_equal(props.get_discharge_rate(train, timestamps), [0, 0])


def test_number_of_spikes(spike_train):
    np.testing.assert_array_equal(props.get_number_of_spikes(spike_train), spike_train.sum(axis=0))


def test_inst_discharge_rate(spike_train, make_train, fs):
    inst_dr = props.get_inst_discharge_rate(spike_train, fs=fs)
    assert inst_dr.shape == spike_train.shape
    # Away from the edges the Hann-weighted 1 s window approximates the rate
    mid = slice(fs // 2 + 200, -fs // 2 - 200)
    np.testing.assert_allclose(inst_dr[mid].mean(axis=0), [10, 20], rtol=0.1)
    assert props.get_inst_discharge_rate(make_train(10), fs=fs).shape[-1] == 1


def test_cov(spike_train, timestamps):
    trains = np.column_stack([spike_train, np.zeros(len(timestamps))])
    cov = props.get_coefficient_of_variation(trains, timestamps)
    np.testing.assert_allclose(cov[:2], 0, atol=1e-10)  # regular firing
    assert np.isnan(cov[2])  # no spikes


def test_cov_discard_isi(make_train, timestamps):
    train = make_train(10)
    train[len(train) // 3 : 2 * len(train) // 3] = 0  # long silent gap
    discard = props.get_coefficient_of_variation(train, timestamps, discard_isi=0.25)
    keep = props.get_coefficient_of_variation(train, timestamps, discard_isi=None)
    assert discard[0] == pytest.approx(0, abs=1e-10)
    assert keep[0] > 0.5


# ---------------------------------------------------------------------------
# IPT-based quality metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def ipts(spike_train, rng):
    clean = spike_train * 10.0 + 0.1 * np.abs(rng.standard_normal(spike_train.shape))
    noisy = spike_train * 10.0 + 4.0 * np.abs(rng.standard_normal(spike_train.shape))
    return clean, noisy


def test_pnr(spike_train, ipts):
    clean, noisy = ipts
    pnr_clean = props.get_pulse_to_noise_ratio(spike_train, clean)
    assert np.all(pnr_clean > 30)
    assert np.all(pnr_clean > props.get_pulse_to_noise_ratio(spike_train, noisy))
    assert np.all(np.isnan(props.get_pulse_to_noise_ratio(np.zeros_like(spike_train), clean)))


def test_silhouette(spike_train, ipts):
    clean, noisy = ipts
    sil_clean = props.get_silhouette_measure(spike_train, clean)
    sil_noisy = props.get_silhouette_measure(spike_train, noisy)
    assert np.all(sil_clean > 0.99)
    assert np.all((sil_noisy <= sil_clean) & (sil_clean <= 1))
    assert np.all(np.isnan(props.get_silhouette_measure(np.zeros_like(spike_train), clean)))


def test_spike_baseline_amp(spike_train):
    spikes_amp, base_amp = props.get_spike_baseline_amp(spike_train, spike_train * 5.0)
    np.testing.assert_allclose(spikes_amp, 5.0)
    np.testing.assert_allclose(base_amp, 0.0)


def test_find_reliable_units():
    # Unit 0 passes; units 1-5 each fail exactly one threshold
    dr = np.array([10, 2, 10, 10, 10, 50])
    cov = np.array([10, 10, 50, 10, 10, 10])
    sil = np.array([0.95, 0.95, 0.95, 0.5, 0.95, 0.95])
    pnr = np.array([35, 35, 35, 35, 20, 35])
    np.testing.assert_array_equal(
        props.find_reliable_units(dr, cov, sil, pnr), [True, False, False, False, False, False]
    )


# ---------------------------------------------------------------------------
# MUAP extraction and centring
# ---------------------------------------------------------------------------


def test_get_muaps(spike_train, emg, muap, fs):
    win_ms = 1000 * muap.shape[-1] / fs  # window matching the template length
    muaps = props.get_muaps(spike_train, emg, fs=fs, win_ms=win_ms)
    assert muaps.shape == (2, *muap.shape)
    np.testing.assert_allclose(muaps[0], muap, atol=1e-12)
    assert props.get_muaps(np.empty(0), emg).shape[0] == 0


def _random_grid_muap(rng, rows, cols, samples, peak_ch, peak_sample):
    """Background channels peaking at random samples, one dominant channel."""
    t = np.arange(samples)
    muap = np.empty((rows, cols, samples))
    for r in range(rows):
        for c in range(cols):
            centre = rng.integers(3, samples - 3)
            muap[r, c] = rng.uniform(0.05, 0.5) * np.exp(-0.5 * ((t - centre) / 2) ** 2)
    muap[peak_ch] = 5.0 * np.exp(-0.5 * ((t - peak_sample) / 2) ** 2)
    return muap


# A small grid (peak samples exceed the channel count) and HD grids (every peak
# sample is also a valid channel index), which the old implementation got wrong
@pytest.mark.parametrize(("rows", "cols", "samples"), [(4, 4, 50), (13, 5, 51), (8, 8, 64)])
def test_center_muaps(rng, rows, cols, samples):
    centre = samples // 2
    for _ in range(25):
        peak_ch = (int(rng.integers(rows)), int(rng.integers(cols)))
        peak_sample = int(rng.integers(3, samples - 3))
        muap = _random_grid_muap(rng, rows, cols, samples, peak_ch, peak_sample)
        centred = props.center_muaps(muap[None])
        # All channels shift together so the dominant peak lands at the centre
        np.testing.assert_array_equal(centred[0], np.roll(muap, centre - peak_sample, axis=-1))


def test_center_muaps_units_independently(make_muap_fn):
    muaps = np.stack([
        make_muap_fn(peak_ch=(0, 0), center=10),
        -make_muap_fn(peak_ch=(3, 1), center=40),  # negative peak
    ])
    original = muaps.copy()
    centred = props.center_muaps(muaps)
    assert np.argmax(np.abs(centred[0, 0, 0])) == 25
    assert np.argmax(np.abs(centred[1, 3, 1])) == 25
    np.testing.assert_array_equal(muaps, original)  # input not modified


def test_center_muaps_shapes(make_muap_fn):
    assert props.center_muaps(make_muap_fn(center=10)).shape == (1, 4, 4, 50)
    assert props.center_muaps(np.empty((0, 4, 4, 50))).shape == (0, 4, 4, 50)


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


@pytest.mark.parametrize(("func", "ratio"), AMPLITUDE_FEATURES)
def test_amplitude_features(func, ratio, muap):
    pair = np.stack([muap, 2 * muap])
    out = func(pair, sel_chs_by=None)
    assert out.shape == pair.shape[:3]
    np.testing.assert_allclose(out[1], ratio * out[0])
    assert func(muap, sel_chs_by=None).shape == (1, *muap.shape[:2])  # single MUAP


@pytest.mark.parametrize(("func", "ratio"), AMPLITUDE_FEATURES)
def test_amplitude_features_channel_selection(func, ratio, muap):
    """Only the dominant channel is selected; the rest are NaN."""
    out = func(np.stack([muap, 2 * muap]), sel_chs_by="iqr")
    assert np.isnan(out).sum() == out.size - 2
    assert out[1, 1, 2] == pytest.approx(ratio * out[0, 1, 2])


@pytest.mark.parametrize("func", [props.get_muap_peak_frequency, props.get_muap_mean_frequency])
def test_frequency_features_pure_tone(func, fs):
    samples = 64
    freq_hz = 8 * fs / samples  # exact FFT bin
    tone = np.tile(np.cos(2 * np.pi * freq_hz * np.arange(samples) / fs), (1, 2, 2, 1))
    np.testing.assert_allclose(func(tone, sel_chs_by=None, fs=fs), freq_hz, rtol=0.05)


@pytest.mark.parametrize("func", FREQUENCY_FEATURES)
def test_frequency_features_narrower_pulse_is_higher(func, pulse, fs):
    wide = np.tile(pulse(25, width=6.0), (1, 2, 2, 1))
    narrow = np.tile(pulse(25, width=1.0), (1, 2, 2, 1))
    assert np.all(func(narrow, sel_chs_by=None, fs=fs) >= func(wide, sel_chs_by=None, fs=fs))


@pytest.mark.parametrize("func", FREQUENCY_FEATURES)
def test_frequency_features_channel_selection(func, pulse, fs):
    """Selection masks the other channels without changing the selected value."""
    muaps = np.tile(0.1 * pulse(25, width=3.0), (1, 4, 4, 1))
    muaps[0, 1, 2] = pulse(25, amp=5.0, width=1.5)
    selected = func(muaps, sel_chs_by="iqr", fs=fs)
    assert np.isnan(selected).sum() == selected.size - 1
    assert selected[0, 1, 2] == func(muaps, sel_chs_by=None, fs=fs)[0, 1, 2]


@pytest.mark.parametrize("func", FREQUENCY_FEATURES)
def test_frequency_features_shift_invariant(func, pulse, fs):
    """The power spectrum is |FFT|^2, so a circular time shift changes nothing."""
    muaps = np.tile(pulse(25, width=2.0) - pulse(30, width=3.0), (1, 2, 2, 1))
    np.testing.assert_allclose(
        func(muaps, sel_chs_by=None, fs=fs),
        func(np.roll(muaps, 7, axis=-1), sel_chs_by=None, fs=fs),
    )


@pytest.mark.parametrize("func", FREQUENCY_FEATURES)
def test_frequency_features_single_muap_matches_batch(func, pulse, fs):
    """A single MUAP is not rectified, so it matches a batch of one."""
    muap = np.tile(pulse(22, amp=5.0) - pulse(28, amp=4.0, width=3.0), (4, 4, 1))
    np.testing.assert_array_equal(
        func(muap, sel_chs_by=None, fs=fs), func(muap[None], sel_chs_by=None, fs=fs)
    )
