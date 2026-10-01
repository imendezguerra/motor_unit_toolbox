import matplotlib.pyplot as plt
import numpy as np
import pytest

from motor_unit_toolbox import plots

pytestmark = pytest.mark.plot


@pytest.mark.parametrize("firings_sorted", [False, True])
def test_plot_spike_trains(spike_train, timestamps, firings_sorted):
    firings = [np.nonzero(spike_train[:, u])[0] for u in range(spike_train.shape[1])]
    _, ax = plt.subplots()
    out = plots.plot_spike_trains(firings, timestamps, firings_sorted=firings_sorted, ax=ax)
    assert out is ax
    assert ax.get_xlim() == (timestamps[0], timestamps[-1])
    assert len(ax.collections) == len(firings)


@pytest.mark.parametrize("ch_framed", ["iqr", "iqr_ptp", "max_amp", "ptp", "per", None])
def test_plot_muaps(muaps, ch_framed):
    original = muaps.copy()
    ax = plots.plot_muaps(muaps, ch_framed=ch_framed)
    assert [a.get_title() for a in ax] == [f"Motor unit: {u}" for u in range(muaps.shape[0])]
    np.testing.assert_array_equal(muaps, original)  # input not modified


def test_plot_muaps_single_unit_normalized(muap):
    assert len(plots.plot_muaps(muap, normalize=True)) == 1


def test_legend_without_duplicate_labels():
    _, ax = plt.subplots()
    for label in ["a", "a", "b"]:
        ax.plot([0, 1], label=label)
    plots.legend_without_duplicate_labels(ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


@pytest.fixture
def clustered(muaps):
    """muaps, cluster_labels, lags, color_labels for two clusters."""
    return muaps, np.array([1, 1, 2]), np.zeros((3, 3), dtype=int), np.array(["a", "b", "a"])


def test_plot_clustered_muaps_order(muaps):
    ax = plots.plot_clustered_muaps(
        muaps, np.array([3, 1, 2]), np.zeros((3, 3), dtype=int), np.array(["a", "b", "c"])
    )
    assert [a.get_title() for a in ax] == ["Cluster: 1", "Cluster: 2", "Cluster: 3"]
    # Each panel holds one unit: one line per channel + time and amplitude scale bars
    rows, cols = muaps.shape[1:3]
    assert [len(a.lines) for a in ax] == [rows * cols + 2] * 3


def test_plot_clustered_muaps_given_axes(clustered, muaps):
    _, axs = plt.subplots(2, 1)
    plots.plot_clustered_muaps(*clustered, ax=axs)
    assert [a.get_title() for a in axs] == ["Cluster: 1", "Cluster: 2"]

    # A single Axes is valid for one cluster only
    _, ax = plt.subplots()
    plots.plot_clustered_muaps(
        muaps[:2], np.array([1, 1]), np.zeros((2, 2), dtype=int), np.array(["a", "b"]), ax=ax
    )
    assert ax.get_title() == "Cluster: 1"
    with pytest.raises(ValueError, match="2 clusters need 2 axes"):
        plots.plot_clustered_muaps(*clustered, ax=ax)


def test_plot_clustered_muaps_per_framing_uses_cluster_muap(clustered):
    """Cluster 2 holds only unit 2 (dominant channel (3, 3)), so its single
    frame must be at row 3, column 3."""
    muaps = clustered[0]
    ax = plots.plot_clustered_muaps(*clustered, ch_framed="per")
    samples = muaps.shape[-1]
    x_step = samples + round(samples / 10)  # column spacing used by the plot
    y_offset = 2.01  # row spacing for MUAPs normalised to a peak of 1
    frames = {
        (round(-(y0 + y_offset / 2) / y_offset), round(x0 / x_step))
        for coll in ax[1].collections
        for x0, y0 in (path.vertices.min(axis=0) for path in coll.get_paths())
    }
    assert frames == {(3, 3)}
