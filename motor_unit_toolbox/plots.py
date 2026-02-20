"""Functions to plot motor unit spike trains and action potentials"""

from copy import copy
from typing import List, Optional, Union
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from motor_unit_toolbox.muap_comp import (
    get_percentile_ch,
    get_highest_amp_ch,
    get_highest_ptp_ch,
    get_highest_iqr_ch,
    get_highest_iqr_ptp_ch,
)


def plot_spike_trains(
    firings: List[List],
    timestamps: Union[list, np.ndarray],
    fs: Optional[int] = 2048,
    firings_sorted: Optional[bool] = False,
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "viridis",
    offset: Optional[float] = 1.5,
    ylabel_offset: Optional[float] = 1,
) -> plt.Axes:
    """Plot spike trains of motor units.

    Args:
        firings (List[List]): List of spike times for each motor unit.
        timestamps (Union[list, np.ndarray]): Array of timestamps.
        fs (Optional[int], optional): Sampling frequency. Defaults to 2048.
        firings_sorted (Optional[bool], optional): Flag to sort the spike
            trains based on recruitment order. Defaults to False.
        ax (Optional[plt.Axes], optional): Axes object to plot the spike trains
            on. If not provided, a new figure and axes will be created.
        palette_name (Optional[str], optional): Name of the color palette to
            use for plotting. Defaults to "viridis".
        offset (Optional[float], optional): Offset between spike trains.
            Defaults to 1.5.
        ylabel_offset (Optional[float], optional): Offset between y-axis labels
            of motor unit indexes. Defaults to 1.

    Returns:
        plt.Axes: Axes object containing the spike train plot.
    """

    if ax is None:
        _, ax = plt.subplots()

    color_palette = sns.color_palette(palette_name, len(firings))

    n_units = len(firings)

    # Plot based on recruitment order based on sorted flag
    sorted_idx = np.arange(0, n_units).astype(int)
    if firings_sorted is True:
        first_firings = np.array([firings[unit][0] for unit in range(n_units)])
        sorted_idx = np.argsort(first_firings)

    ax.eventplot(
        [firings[idx]/fs + timestamps[0] for idx in sorted_idx],
        orientation="horizontal",
        colors=color_palette,
        lineoffsets=offset,
    )
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_yticks(
        np.arange(0, n_units * offset, offset * ylabel_offset),
        sorted_idx.astype(int)[::ylabel_offset],
    )
    ax.set_ylabel("Motor unit indexes")
    ax.set_xlabel("Time (s)")
    ax.set_ylim([-offset, n_units * offset + offset])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax


def plot_muaps(
    muaps: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "viridis",
    normalize: Optional[bool] = False,
    ch_framed: Optional[str] = "iqr",
) -> plt.Axes:
    """Plot motor unit action potentials (MUAPs).

    Args:
        muaps (np.ndarray): Motor unit action potentials, with shape
            (units, ch_rows, ch_cols, samples).
        fs (Optional[int], optional): Sampling frequency. Defaults to 2048.
        ax (Optional[plt.Axes], optional): Axes object to plot the MUAPs on.
            If not provided, a new figure and axes will be created.
        palette_name (Optional[str], optional): Name of the color palette to
            use for plotting. Defaults to "viridis".
        normalize (Optional[bool], optional): Flag to normalize the MUAPs.
            Defaults to False.
        ch_framed (Optional[str], optional): Channel framing method. Defaults
            to "iqr".

    Returns:
        plt.Axes: Axes object containing the MUAP plot.
    """

    # Check muap dimensions
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)

    # Normalise muaps for plots
    muaps_plot = copy(muaps)
    if normalize:
        max_amp = np.nanmax(np.abs(muaps), axis=(1, 2, 3))
        muaps_plot /= max_amp[:, None, None, None]
    else:
        max_amp = np.nanmax(np.abs(muaps))
        muaps_plot /= max_amp

    #  Initialise variables
    n_units, rows, cols, samples = muaps_plot.shape
    x_offset = round(muaps_plot.shape[-1] / 10)
    y_offset = np.nanmax(np.abs(muaps_plot)) * 2.01
    x = np.arange(samples)

    # Define axes if not initialised
    if ax is None:
        _, ax = plt.subplots(n_units, 1, figsize=(20, 5 * n_units))
        ax = np.ravel(ax)

    # Create color palette
    color_palette = sns.color_palette(palette_name, n_units)

    # Plot muaps_plot
    for unit in range(n_units):

        # Get highest amplitude channels
        if ch_framed == "max_amp":
            curr_amp_ch = get_highest_amp_ch(muaps_plot[unit])
        elif ch_framed == "ptp":
            curr_amp_ch = get_highest_ptp_ch(muaps_plot[unit])
        elif ch_framed == "iqr":
            curr_amp_ch = get_highest_iqr_ch(muaps_plot[unit])
        elif ch_framed == "iqr_ptp":
            curr_amp_ch = get_highest_iqr_ptp_ch(muaps_plot[unit])
        elif ch_framed == "per":
            curr_amp_ch = get_percentile_ch(muaps_plot[unit], thr=95)

        for col in range(cols):
            # Apply offset to signals
            curr_muap_col = muaps_plot[unit, :, col].T
            curr_muap_col -= range(0, rows) * y_offset

            # Current x
            curr_x = x + (samples + x_offset) * col

            # Plot column
            ax[unit].plot(curr_x, curr_muap_col, color=color_palette[unit], linewidth=1)

            # Plot frames
            if ch_framed is None:
                continue
            frames = []
            for row in range(rows):
                if curr_amp_ch[row, col] == 0:
                    continue
                frames.append(
                    Rectangle(
                        (curr_x[0] - x_offset / 2, -y_offset * row - y_offset / 2),
                        width=samples + x_offset,
                        height=y_offset,
                    )
                )
            frame_collection = PatchCollection(
                frames, ls="-", ec="lightgrey", fc="none", lw=1
            )
            ax[unit].add_collection(frame_collection)

        # Add time reference
        time_y_ref = -y_offset * (rows) - y_offset / 10
        ax[unit].plot([0, samples], [time_y_ref, time_y_ref], "-", color="black")
        ax[unit].annotate(
            f"{samples/fs*1000:.0f} ms",
            xy=(samples / 2, time_y_ref),
            xytext=(0, time_y_ref + y_offset / 10),
        )

        # Add amplitude reference
        amp_x_ref = (samples + x_offset) * 2
        max_ptp = np.nanmax(np.ptp(muaps_plot[unit], axis=-1))
        amp_y_ref = [-max_ptp / 2, max_ptp / 2] - y_offset * rows - y_offset / 10
        ax[unit].plot([amp_x_ref, amp_x_ref], amp_y_ref, "-", color="black")
        ax[unit].annotate(
            f"{np.max(np.ptp(muaps[unit], axis=-1)):.2f} mV",
            xy=(amp_x_ref, np.mean(amp_y_ref)),
            xytext=(amp_x_ref * 1.1, np.mean(amp_y_ref)),
        )

        # Remove axes
        ax[unit].set_xlim([-x_offset, (x_offset + samples) * cols])
        ax[unit].set_ylim([-y_offset * (rows + 1), y_offset])
        ax[unit].set_title(f"Motor unit: {unit}")
        ax[unit].set_axis_off()

    for i in range(unit + 1, len(ax)):
        ax[i].set_axis_off()

    return ax


def legend_without_duplicate_labels(ax: plt.Axes) -> None:
    """
    Create a legend without duplicate labels.

    Args:
        ax (plt.Axes): Axes object to create the legend on.
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique), bbox_to_anchor=(1.2, 1))


def plot_clustered_muaps(
    muaps: np.ndarray,
    cluster_labels,
    lags: np.ndarray,
    color_labels: np.ndarray,
    fs: Optional[int] = 2048,
    ax: Optional[plt.Axes] = None,
    palette_name: Optional[str] = "viridis",
    color_order: Optional[list] = None,
    normalize: Optional[bool] = False,
    ch_framed: Optional[str] = "iqr",
) -> plt.Axes:
    """Plot clustered motor unit action potentials (MUAPs).

    Args:
        muaps (np.ndarray): Array of MUAPs with shape (n_units, rows, cols,
            samples).
        cluster_labels (np.ndarray): Array of cluster labels for each MUAP.
        lags (np.ndarray): Array of temporal alignment lags for each MUAP.
        color_labels (np.ndarray): Array of color labels for each MUAP.
        fs (Optional[int], optional): Sampling frequency in Hz. Defaults to
            2048.
        ax (Optional[plt.Axes], optional): Axes object to plot on. Defaults to
            None.
        palette_name (Optional[str], optional): Name of the color palette.
            Defaults to "viridis".
        color_order (Optional[list], optional): Order of color labels in the
            legend. Defaults to None.
        normalize (Optional[bool], optional): Flag indicating whether to
            normalize the MUAPs. Defaults to False.
        ch_framed (Optional[str], optional): Channel framing method. Defaults
            to 'iqr'.

    Returns:
        plt.Axes: Axes object with the plotted MUAPs.
    """
    
    # Check muaps dimensions
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)

    # Normalise muaps for plots
    muaps_plot = copy(muaps)
    if normalize:
        max_amp = np.nanmax(muaps, axis=(1, 2, 3))
        muaps_plot /= max_amp[:, None, None, None]
    else:
        max_amp = np.nanmax(np.abs(muaps))
        muaps_plot /= max_amp

    #  Initialise variables
    _, rows, cols, samples = muaps_plot.shape
    x_offset = round(muaps_plot.shape[-1] / 10)
    y_offset = np.nanmax(np.abs(muaps_plot)) * 2.01
    x = np.arange(samples)
    clusters = np.unique(cluster_labels)
    n_clusters = len(clusters)

    # Create color palette
    if color_order is None:
        color_order = np.unique(color_labels)
    color_palette = sns.color_palette(palette_name, len(color_order))
    color_dict = {label: color_palette[i] for i, label in enumerate(color_order)}

    # Define axes if not initialised
    if ax is None:
        _, ax = plt.subplots(n_clusters, 1, figsize=(10, 4 * n_clusters))
        ax = np.ravel(ax)
    else:
        ax = np.expand_dims(ax, axis=0)

    for c, cluster in enumerate(clusters):

        # Get all the units belonging to the cluster
        cluster_idx = np.nonzero(cluster_labels == cluster)[0]
        muaps_cluster = muaps_plot[cluster_idx]
        cluster_color_labels = color_labels[cluster_idx]
        cluster_lags = lags[cluster_idx[0], cluster_idx]

        if normalize is False:
            # Normalise cluster amplitude
            cluster_max_amp = np.nanmax(np.abs(muaps_cluster))
            muaps_cluster /= cluster_max_amp

        # Check size
        if len(muaps_cluster.shape) < 4:
            muaps_cluster = np.expand_dims(muaps_cluster, axis=0)
        n_units_cluster = muaps_cluster.shape[0]

        # Plot muaps_plot
        for unit in range(n_units_cluster):

            # Apply temporal alignment
            muaps_cluster[unit] = np.roll(
                muaps_cluster[unit], cluster_lags[unit], axis=-1
            )

            # Get highest amplitude channels
            if ch_framed == "max_amp":
                curr_amp_ch = get_highest_amp_ch(muaps_cluster[unit])
            elif ch_framed == "ptp":
                curr_amp_ch = get_highest_ptp_ch(muaps_cluster[unit])
            elif ch_framed == "iqr":
                curr_amp_ch = get_highest_iqr_ch(muaps_cluster[unit])
            elif ch_framed == "iqr_ptp":
                curr_amp_ch = get_highest_iqr_ptp_ch(muaps_cluster[unit])
            elif ch_framed == "per":
                curr_amp_ch = get_percentile_ch(muaps_plot[unit], thr=95)

            for col in range(cols):
                # Apply offset to signals
                curr_muap_col = muaps_cluster[unit, :, col].T
                curr_muap_col -= range(0, rows) * y_offset

                # Current x
                curr_x = x + (samples + x_offset) * col

                # Plot column
                if col == 0:
                    curr_label = cluster_color_labels[unit]
                else:
                    curr_label = None
                ax[c - 1].plot(
                    curr_x,
                    curr_muap_col,
                    color=color_dict[cluster_color_labels[unit]],
                    linewidth=1,
                    label=curr_label,
                )

                # Plot frames
                if ch_framed is None:
                    continue
                frames = []
                for row in range(rows):
                    if curr_amp_ch[row, col] == 0:
                        continue
                    frames.append(
                        Rectangle(
                            (curr_x[0] - x_offset / 2, -y_offset * row - y_offset / 2),
                            width=samples + x_offset,
                            height=y_offset,
                        )
                    )
                frame_collection = PatchCollection(
                    frames, ls="-", ec="lightgrey", fc="none", lw=1
                )
                ax[c - 1].add_collection(frame_collection)

        # Add time reference
        time_y_ref = -y_offset * (rows) - y_offset / 10
        ax[c - 1].plot([0, samples], [time_y_ref, time_y_ref], "-", color="black")
        ax[c - 1].annotate(
            f"{samples/fs*1000:.0f} ms",
            xy=(samples / 2, time_y_ref),
            xytext=(0, time_y_ref + y_offset / 10),
        )

        # Add amplitude reference
        amp_x_ref = (samples + x_offset) * 2
        max_ptp = np.nanmax(np.ptp(muaps_cluster, axis=-1))
        amp_y_ref = [-max_ptp / 2, max_ptp / 2] - y_offset * rows - y_offset / 10
        ax[c - 1].plot([amp_x_ref, amp_x_ref], amp_y_ref, "-", color="black")
        ax[c - 1].annotate(
            f"{np.nanmax(np.ptp(muaps[cluster_idx], axis=-1)):.2f} mV",
            xy=(amp_x_ref, np.mean(amp_y_ref)),
            xytext=(amp_x_ref * 1.1, np.mean(amp_y_ref)),
        )

        # Remove axes
        ax[c - 1].set_xlim([-x_offset, (x_offset + samples) * cols])
        ax[c - 1].set_ylim([-y_offset * (rows + 1), y_offset])
        ax[c - 1].set_title(f"Cluster: {cluster}")
        legend_without_duplicate_labels(ax[c - 1])
        ax[c - 1].set_axis_off()

    return ax
