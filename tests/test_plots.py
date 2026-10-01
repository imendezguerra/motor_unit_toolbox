import matplotlib.pyplot as plt
import numpy as np
import pytest

from motor_unit_toolbox import plots

pytestmark = pytest.mark.plot


@pytest.fixture
def firings(spike_train):
    return [np.nonzero(spike_train[:, u])[0] for u in range(spike_train.shape[1])]


@pytest.mark.parametrize("firings_sorted", [False, True])
def test_plot_spike_trains(firings, timestamps, firings_sorted):
    ax = plots.plot_spike_trains(firings, timestamps, firings_sorted=firings_sorted)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlim() == (timestamps[0], timestamps[-1])
    assert len(ax.collections) == len(firings)


def test_plot_spike_trains_on_given_axes(firings, timestamps):
    _, ax = plt.subplots()
    assert plots.plot_spike_trains(firings, timestamps, ax=ax) is ax


@pytest.mark.parametrize("ch_framed", ["iqr", "iqr_ptp", "max_amp", "ptp", "per", None])
@pytest.mark.parametrize("normalize", [False, True])
def test_plot_muaps(muaps, ch_framed, normalize):
    original = muaps.copy()
    ax = plots.plot_muaps(muaps, ch_framed=ch_framed, normalize=normalize)
    assert len(ax) == muaps.shape[0]
    assert [a.get_title() for a in ax] == [f"Motor unit: {u}" for u in range(muaps.shape[0])]
    np.testing.assert_array_equal(muaps, original)  # input not mutated


def test_plot_muaps_single_unit(muap):
    ax = plots.plot_muaps(muap)
    assert len(ax) == 1


def test_legend_without_duplicate_labels():
    _, ax = plt.subplots()
    ax.plot([0, 1], label="a")
    ax.plot([1, 0], label="a")
    ax.plot([0, 0], label="b")
    plots.legend_without_duplicate_labels(ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


@pytest.fixture
def clustered(muaps):
    cluster_labels = np.array([1, 1, 2])
    lags = np.zeros((3, 3), dtype=int)
    color_labels = np.array(["trial0", "trial1", "trial0"])
    return muaps, cluster_labels, lags, color_labels


def test_plot_clustered_muaps_runs(clustered):
    ax = plots.plot_clustered_muaps(*clustered)
    assert len(ax) == 2


@pytest.mark.xfail(
    raises=AssertionError,
    reason="plot_clustered_muaps draws cluster c on ax[c - 1], so titles are rotated",
)
def test_plot_clustered_muaps_axis_order(clustered):
    ax = plots.plot_clustered_muaps(*clustered)
    assert [a.get_title() for a in ax] == ["Cluster: 1", "Cluster: 2"]
