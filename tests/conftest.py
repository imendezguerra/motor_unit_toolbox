"""Shared fixtures: small, deterministic synthetic motor unit data."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

FS = 2048
DURATION_S = 2
N_SAMPLES = FS * DURATION_S
FIRING_RATES_HZ = (10, 20)
GRID = (4, 4)
MUAP_SAMPLES = 50


def regular_spike_train(rate_hz, n_samples=N_SAMPLES, fs=FS, offset=100):
    """Binary spike train firing at a constant rate."""
    train = np.zeros(n_samples, dtype=int)
    train[offset::round(fs / rate_hz)] = 1
    return train


def gaussian_pulse(center, amp=1.0, width=2.0, n_samples=MUAP_SAMPLES):
    t = np.arange(n_samples)
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def make_muap(peak_ch=(1, 2), center=25, amp=5.0, bg_amp=0.1, grid=GRID):
    """MUAP of shape (rows, cols, samples) with one dominant channel."""
    rows, cols = grid
    muap = np.zeros((rows, cols, MUAP_SAMPLES))
    for r in range(rows):
        for c in range(cols):
            muap[r, c] = gaussian_pulse(center, bg_amp)
    muap[peak_ch] = gaussian_pulse(center, amp)
    return muap


@pytest.fixture
def make_train():
    """Factory for regular binary spike trains (see ``regular_spike_train``)."""
    return regular_spike_train


@pytest.fixture
def make_muap_fn():
    """Factory for synthetic MUAPs (see ``make_muap``)."""
    return make_muap


@pytest.fixture
def pulse():
    """Factory for Gaussian pulses (see ``gaussian_pulse``)."""
    return gaussian_pulse


@pytest.fixture
def fs():
    return FS


@pytest.fixture
def timestamps():
    return np.arange(N_SAMPLES) / FS


@pytest.fixture
def spike_train():
    """(n_samples, 2) binary spike trains at 10 Hz and 20 Hz."""
    return np.stack([regular_spike_train(r) for r in FIRING_RATES_HZ], axis=-1)


@pytest.fixture
def muap():
    return make_muap()


@pytest.fixture
def muaps():
    """(3, rows, cols, samples): three distinct units on different channels."""
    return np.stack([
        make_muap(peak_ch=(0, 0)),
        make_muap(peak_ch=(1, 2)),
        make_muap(peak_ch=(3, 3)),
    ])


@pytest.fixture
def emg(spike_train):
    """(rows, cols, n_samples) EMG generated from unit 0 with a known MUAP."""
    muap = make_muap()
    rows, cols, _ = muap.shape
    emg = np.zeros((rows, cols, N_SAMPLES))
    half = MUAP_SAMPLES // 2
    for firing in np.nonzero(spike_train[:, 0])[0]:
        start, stop = firing - half, firing + half
        if start >= 0 and stop <= N_SAMPLES:
            emg[:, :, start:stop] += muap
    return emg


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
