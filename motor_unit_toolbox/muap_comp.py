"""Functions to compute MUAP similarity and distance metrics"""

import itertools
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from scipy import cluster, optimize, signal, spatial, stats
from sklearn import metrics


def get_highest_iqr_ch(muap: np.ndarray) -> np.ndarray:
    """Compute the highest interquartile range (IQR) channels.

    Args:
        muap (ndarray): Motor unit action potential with shape
            (ch_rows, ch_cols, samples)

    Returns:
        ndarray: A boolean mask indicating the channels with the highest IQR.
    """

    peak_amps = np.amax(np.abs(muap), axis=-1)
    peak_amps = np.reshape(peak_amps, (-1))
    q1 = np.percentile(peak_amps, 25)
    q3 = np.percentile(peak_amps, 75)
    iqr = q3 - q1
    whis = 1.5
    outliers = peak_amps > q3 + whis * iqr
    ch_mask = np.reshape(outliers, (muap.shape[0], muap.shape[1]))
    return ch_mask


def get_percentile_ch(
    muap: np.ndarray,
    thr: Optional[int] = 90
):
    """Compute the channels that exceed a certain percentile threshold.

    Args:
        muap (ndarray): Motor unit action potential with shape
            (ch_rows, ch_cols, samples)
        thr (int, optional): The percentile threshold. Defaults to 90.

    Returns:
        ndarray: A boolean mask indicating the channels that exceed the
            specified percentile threshold.
    """
    peak_amps = np.amax(np.abs(muap), axis=-1)
    peak_amps = np.reshape(peak_amps, (-1))
    outliers = peak_amps > np.percentile(peak_amps, thr)
    ch_mask = np.reshape(outliers, (muap.shape[0], muap.shape[1]))
    return ch_mask


def get_highest_iqr_ptp_ch(muap: np.ndarray) -> np.ndarray:
    """Compute the channels with the highest peak-to-peak (PTP) amplitude.

    Args:
        muap (np.ndarray): Motor unit action potential with shape
            (ch_rows, ch_cols, samples)

    Returns:
        np.ndarray: A boolean mask indicating the channels with the highest
            PTP amplitude.
    """

    peak_amps = np.ptp(muap, axis=-1)
    peak_amps = np.reshape(peak_amps, (-1))
    q1 = np.percentile(peak_amps, 25)
    q3 = np.percentile(peak_amps, 75)
    iqr = q3 - q1
    whis = 1.5
    outliers = peak_amps > q3 + whis * iqr
    ch_mask = np.reshape(outliers, (muap.shape[0], muap.shape[1]))
    return ch_mask


def get_highest_amp_ch(muap: np.ndarray) -> np.ndarray:
    """Compute the channels with the highest amplitude.

    Args:
        muap (ndarray): Motor unit action potential with shape
            (ch_rows, ch_cols, samples)

    Returns:
        ndarray: A boolean mask indicating the channels with the highest
            amplitude.
    """
    peak_amps = np.amax(np.abs(muap), axis=-1)
    amp_thr = 3 * np.std(peak_amps)
    ch_mask = peak_amps > amp_thr
    return ch_mask


def get_highest_ptp_ch(muap: np.ndarray) -> np.ndarray:
    """Compute the channels with the highest peak-to-peak (PTP) amplitude.

    Args:
        muap (np.ndarray): Motor unit action potential with shape
            (ch_rows, ch_cols, samples)

    Returns:
        np.ndarray: A boolean mask indicating the channels with the highest
            PTP amplitude.
    """
    ptp_amps = np.ptp(muap, axis=-1)
    amp_thr = 2 * np.std(ptp_amps)
    ch_mask = ptp_amps > amp_thr
    return ch_mask


def nmse(muap1: np.ndarray, muap2: np.ndarray) -> float:
    """Compute the normalized mean squared error (NMSE) between two MUAPs.

    Args:
        muap1 (np.ndarray): First motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        muap2 (np.ndarray): Second motor unit action potential with shape
            (ch_rows, ch_cols, samples).

    Returns:
        float: The NMSE between the two MUAPs.
    """
    mse = np.power(np.linalg.norm(muap1 - muap2), 2)
    m_energy = np.mean(
        np.power(np.linalg.norm(muap1), 2) + np.power(np.linalg.norm(muap2), 2)
    )

    return mse / m_energy


def norm_farina_distance(muap1: np.ndarray, muap2: np.ndarray) -> float:
    """Compute the normalized Farina distance between two MUAPs.

    Args:
        muap1 (np.ndarray): First motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        muap2 (np.ndarray): Second motor unit action potential with shape
            (ch_rows, ch_cols, samples).

    Returns:
        float: The normalized Farina distance between the two MUAPs.
    """
    # Check dimensions
    if len(muap1.shape) == 1:
        muap1 = np.expand_dims(muap1, axis=0)
    if len(muap2.shape) == 1:
        muap2 = np.expand_dims(muap2, axis=0)

    # Compute means
    muap1_mean = np.mean(muap1, axis=-1)
    muap2_mean = np.mean(muap2, axis=-1)

    # Compute norms
    muap1_l2 = np.linalg.norm(muap1, 2)
    muap2_l2 = np.linalg.norm(muap2, 2)

    # Normalise muaps
    muap1_norm = (muap1 - muap1_mean[:, None]) / muap1_l2
    muap2_norm = (muap2 - muap2_mean[:, None]) / muap2_l2

    # Compute energy
    muap1_energy = np.sum(np.power(muap1_norm, 2))
    muap2_energy = np.sum(np.power(muap2_norm, 2))

    # Compute normalized farina distance
    muaps_nfd = (2 * np.sum(np.power(muap1_norm - muap2_norm, 2))) / (
        muap1_energy + muap2_energy
    )

    return muaps_nfd


def get_alignmnent(
        muap1: np.ndarray,
        muap2: np.ndarray,
        flag_debug: Optional[bool] = False
        ) -> int:
    """Compute the alignment lag between two MUAPs.

    Args:
        muap1 (np.ndarray): First motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        muap2 (np.ndarray): Second motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        flag_debug (bool, optional): Flag to enable debug mode. Defaults to
            False.

    Returns:
        int: The alignment lag between the two MUAPs.
    """
    # Get the muap length
    muap_length = muap1.shape[-1]

    # Flatten muaps for the correlation
    muap1_flat = muap1.flatten()
    muap2_flat = muap2.flatten()

    # Compute correlation and lags
    corr = signal.correlate(muap1_flat, muap2_flat, mode="full")
    lags = signal.correlation_lags(len(muap1_flat), len(muap2_flat), mode="full")

    # Bound both to half the muap length in each direction
    boundary_mask = np.nonzero(
        np.logical_and(lags >= -muap_length // 2, lags <= muap_length // 2)
    )[0]
    bounded_corr = corr[boundary_mask]
    bounded_lags = lags[boundary_mask]
    muaps_lag = bounded_lags[np.argmax(np.abs(bounded_corr))].astype(int)

    if not np.isscalar(muaps_lag):
        # If there is more than one possible lag, choose the minimum
        muaps_lag = np.amin(muaps_lag)

    if flag_debug:
        _, axs = plt.subplots(3, 1)

        axs[0].plot(bounded_lags, bounded_corr)
        axs[0].plot(muaps_lag, bounded_corr[np.argmax(np.abs(bounded_corr))], "ro")
        axs[0].set_ylabel("Correlation")
        axs[0].set_xlabel("Lags")
        axs[0].grid(True)
        axs[0].set_xlim([bounded_lags[0], bounded_lags[-1]])
        axs[0].set_title(f"max_corr = {np.amax(bounded_corr):.3f} at lag = {muaps_lag}")

        axs[1].plot(muap1_flat)
        axs[1].plot(muap2_flat)
        axs[1].grid(True)
        axs[1].set_xlim([0, len(muap1_flat)])
        axs[1].set_title("Before alignment")

        axs[2].plot(muap1_flat)
        axs[2].plot(np.roll(muap2_flat, muaps_lag, axis=-1))
        axs[2].grid(True)
        axs[2].set_xlim([0, len(muap1_flat)])
        axs[2].set_title(f"After alignment (lag = {muaps_lag})")

        plt.tight_layout()
        plt.show()

    return muaps_lag


def compute_muaps_similarity(
    muap1: np.ndarray,
    muap2: np.ndarray,
    sel_chs_by: Literal["iqr", "iqr_ptp", "max_abs", "ptp"] = "iqr",
    metric: Literal["corr", "cosine", "nmse", "nfd"] = "nmse"
) -> Tuple[float, int]:
    """Compute the similarity between two MUAPs.

    Args:
        muap1 (np.ndarray): First motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        muap2 (np.ndarray): Second motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        sel_chs_by (str, optional): Method for selecting significant
            amplitude channels: "iqr", "iqr_ptp", "max_abs" or "ptp". Defaults to "iqr".
        metric (str, optional): Similarity metric to compute:
            "corr", "cosine", "nmse" or "nfd". Defaults to "nmse".

    Returns:
        Tuple[float, int]: The similarity between the two MUAPs and the
            alignment lag.
    """
    # Get significant amplitude channels
    if sel_chs_by == "max_abs":
        muap1_ch_mask = get_highest_amp_ch(muap1)
        muap2_ch_mask = get_highest_amp_ch(muap2)
    elif sel_chs_by == "ptp":
        muap1_ch_mask = get_highest_ptp_ch(muap1)
        muap2_ch_mask = get_highest_ptp_ch(muap2)
    elif sel_chs_by == "iqr":
        muap1_ch_mask = get_highest_iqr_ch(muap1)
        muap2_ch_mask = get_highest_iqr_ch(muap2)
    elif sel_chs_by == "iqr_ptp":
        muap1_ch_mask = get_highest_iqr_ptp_ch(muap1)
        muap2_ch_mask = get_highest_iqr_ptp_ch(muap2)

    # Check if channel masks are empty
    if not np.any(muap1_ch_mask):
        muap1_ch_mask = np.empty(muap1.shape[0:2])
    if not np.any(muap2_ch_mask):
        muap2_ch_mask = np.empty(muap2.shape[0:2])

    # Apply the channel selection to the muaps and correct for the lag
    sel_chs = np.nonzero(np.logical_or(muap1_ch_mask, muap2_ch_mask))
    if not np.any(sel_chs):
        sel_chs = np.ones_like(muap1).astype(bool)

    muap1_sel = np.array(muap1[sel_chs])
    muap2_sel = np.array(muap2[sel_chs])

    # Align signals based on the selected channels
    muaps_lag = get_alignmnent(muap1_sel, muap2_sel)
    muap2_sel = np.roll(muap2_sel, muaps_lag, axis=-1)

    # Compute metric
    if metric == "corr":
        muaps_sim = stats.pearsonr(muap1_sel.flatten(), muap2_sel.flatten())[0]
    elif metric == "cosine":
        muaps_sim = 1 - spatial.distance.cosine(
            muap1_sel.flatten(), muap2_sel.flatten()
        )
    elif metric == "nmse":
        muaps_sim = 1 - nmse(muap1_sel.flatten(), muap2_sel.flatten())
    elif metric == "nfd":
        muaps_sim = 1 - norm_farina_distance(muap1_sel.flatten(), muap2_sel.flatten())

    return muaps_sim, muaps_lag


def compute_muaps_dist(
        muap1: np.ndarray,
        muap2: np.ndarray,
        sel_chs_by: Literal["iqr", "iqr_ptp", "max_abs", "ptp"] = "iqr",
        metric: Literal["corr", "cosine", "nmse", "nfd"] = "nmse"
        ) -> Tuple[float, int]:
    """Compute the distance between two MUAPs.

    Args:
        muap1 (np.ndarray): First motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        muap2 (np.ndarray): Second motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        sel_chs_by (str, optional): Method for selecting significant
            amplitude channels: "iqr", "iqr_ptp", "max_abs" or "ptp". Defaults to "iqr".
        metric (str, optional): Distance metric to compute:
            "corr", "cosine", "nmse" or "nfd". Defaults to "nmse".

    Returns:
        Tuple[float, int]: The distance between the two MUAPs and the
            alignment lag.
    """

    # Get significant amplitude channels
    if sel_chs_by == "max_abs":
        muap1_ch_mask = get_highest_amp_ch(muap1)
        muap2_ch_mask = get_highest_amp_ch(muap2)
    elif sel_chs_by == "ptp":
        muap1_ch_mask = get_highest_ptp_ch(muap1)
        muap2_ch_mask = get_highest_ptp_ch(muap2)
    elif sel_chs_by == "iqr":
        muap1_ch_mask = get_highest_iqr_ch(muap1)
        muap2_ch_mask = get_highest_iqr_ch(muap2)
    elif sel_chs_by == "iqr_ptp":
        muap1_ch_mask = get_highest_iqr_ptp_ch(muap1)
        muap2_ch_mask = get_highest_iqr_ptp_ch(muap2)

    # Check if channel masks are empty
    if not np.any(muap1_ch_mask):
        muap1_ch_mask = np.empty(muap1.shape[0:2])
    if not np.any(muap2_ch_mask):
        muap2_ch_mask = np.empty(muap2.shape[0:2])

    # Apply the channel selection to the muaps and correct for the lag
    sel_chs = np.nonzero(np.logical_or(muap1_ch_mask, muap2_ch_mask))
    if not np.any(sel_chs):
        sel_chs = np.ones_like(muap1).astype(bool)

    muap1_sel = np.array(muap1[sel_chs])
    muap2_sel = np.array(muap2[sel_chs])

    # Align signals based on the selected channels
    muaps_lag = get_alignmnent(muap1_sel, muap2_sel)
    muap2_sel = np.roll(muap2_sel, muaps_lag, axis=-1)

    # Compute metric
    if metric == "corr":
        muaps_dist = 1 - stats.pearsonr(muap1_sel.flatten(), muap2_sel.flatten())[0]
    elif metric == "cosine":
        muaps_dist = spatial.distance.cosine(muap1_sel.flatten(), muap2_sel.flatten())
    elif metric == "nmse":
        muaps_dist = nmse(muap1_sel.flatten(), muap2_sel.flatten())
    elif metric == "nfd":
        muaps_dist = norm_farina_distance(muap1_sel.flatten(), muap2_sel.flatten())

    return muaps_dist, muaps_lag


def compute_muaps_dist_sets(
    muaps1: np.ndarray,
    muaps2: np.ndarray,
    dist_metric: Literal["corr", "cosine", "nmse", "nfd"] = "nmse",
    sel_chs_method: Literal["iqr", "iqr_ptp", "max_abs", "ptp"] = "iqr"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the distance between sets of MUAPs.

    Args:
        muaps1 (np.ndarray): First set of motor unit action potentials with shape
            (n_units1, ch_rows, ch_cols, samples).
        muaps2 (np.ndarray): Second set of motor unit action potentials with shape
            (n_units2, ch_rows, ch_cols, samples).
        dist_metric (str, optional): Distance metric to compute:
            "corr", "cosine", "nmse" or "nfd". Defaults to "nmse".
        sel_chs_method (str, optional): Method for selecting significant
            amplitude channels: "iqr", "iqr_ptp", "max_abs" or "ptp". Defaults to "iqr".

    Returns:
        Tuple[np.ndarray, np.ndarray]: The distance matrix between the two sets of
            MUAPs and the alignment lag matrix.
    """

    # Check muaps size
    if np.any([len(muaps1) == 0, len(muaps2) == 0]):
        # No units
        all_muaps_dist = np.empty(0)
        all_muaps_lags = np.empty(0)
    else:
        # Check units sizes
        if len(muaps1.shape) < 4:
            muaps1 = np.expand_dims(muaps1, axis=0)

        if len(muaps2.shape) < 4:
            muaps2 = np.expand_dims(muaps2, axis=0)

        # Initialise variables
        n_units1 = muaps1.shape[0]
        n_units2 = muaps2.shape[0]

        # Initialise output
        all_muaps_dist = np.zeros((n_units1, n_units2))
        all_muaps_lags = np.zeros((n_units1, n_units2))

        # Fill correlation matrix for each unique pair comparison
        for unit1, unit2 in itertools.product(range(n_units1), range(n_units2)):

            # Select motor units
            muap1 = muaps1[unit1]
            muap2 = muaps2[unit2]

            # Compute correlation and lags
            curr_muaps_dist, curr_muaps_lag = compute_muaps_dist(
                muap1, muap2, sel_chs_by=sel_chs_method, metric=dist_metric
            )

            # Fill matrices
            all_muaps_dist[unit1, unit2] = curr_muaps_dist
            all_muaps_lags[unit1, unit2] = curr_muaps_lag

    return all_muaps_dist, all_muaps_lags.astype(int)


def compute_all_muaps_dist(
    muaps: np.ndarray,
    dist_metric: Literal["corr", "cosine", "nmse", "nfd"] = "nmse",
    sel_chs_method: Literal["iqr", "iqr_ptp", "max_abs", "ptp"] = "iqr"
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the distance between all pairs of MUAPs.

    Args:
        muaps (np.ndarray): Motor unit action potentials with shape
            (n_units, ch_rows, ch_cols, samples).
        dist_metric (str, optional): Distance metric to compute:
            "corr", "cosine", "nmse" or "nfd". Defaults to "nmse".
        sel_chs_method (str, optional): Method for selecting significant
            amplitude channels: "iqr", "iqr_ptp", "max_abs" or "ptp". Defaults to "iqr".

    Returns:
        Tuple[np.ndarray, np.ndarray]: The distance matrix between all pairs of MUAPs
            and the alignment lag matrix.
    """

    # Check muaps size
    if len(muaps) == 0:
        # No units
        all_muaps_dist = np.empty(0)
        all_muaps_lags = np.empty(0)
    else:
        # There are units
        if len(muaps.shape) < 4:
            muaps = np.expand_dims(muaps, axis=0)

        # Initialise variables
        n_units = muaps.shape[0]

        if n_units == 1:
            # Only one unit
            all_muaps_dist = np.array([0])
            all_muaps_lags = np.array([0])
        else:
            # More than one unit, so initialise pair combinations
            pairs = itertools.combinations(range(n_units), 2)
            all_muaps_dist = np.zeros((n_units, n_units))
            all_muaps_lags = np.zeros((n_units, n_units))

            # Fill correlation matrix for each unique pair comparison
            for pair in pairs:
                # Select motor units
                muap1 = muaps[pair[0]]
                muap2 = muaps[pair[1]]

                # Compute correlation and lags
                curr_muaps_dist, curr_muaps_lag = compute_muaps_dist(
                    muap1, muap2, sel_chs_by=sel_chs_method, metric=dist_metric
                )

                # Fill matrices
                all_muaps_dist[pair[0], pair[1]] = curr_muaps_dist
                all_muaps_dist[pair[1], pair[0]] = curr_muaps_dist

                all_muaps_lags[pair[0], pair[1]] = curr_muaps_lag
                all_muaps_lags[pair[1], pair[0]] = curr_muaps_lag

    return all_muaps_dist, all_muaps_lags.astype(int)


def cluster_muaps(
    all_muaps_dist: np.ndarray,
    cluster_method: Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ] = "ward",
    dist_metric: Literal["corr", "cosine", "nmse", "nfd"] = "corr",
    sel_chs_method: Literal["iqr", "iqr_ptp", "max_abs", "ptp"] = "max_abs",
    thr_vals: Optional[np.ndarray] = np.arange(0, 2.0001, 0.001),  # noqa: B008 (read-only)
    flag_plot: Optional[bool] = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Cluster MUAPs based on distance matrix.

    Args:
        all_muaps_dist (np.ndarray): Distance matrix between all pairs of MUAPs.
        cluster_method (str, optional): Linkage method passed to
            scipy.cluster.hierarchy.linkage. Defaults to "ward".
        dist_metric (str, optional): Distance metric used to build
            all_muaps_dist (stored in the output only). Defaults to "corr".
        sel_chs_method (str, optional): Channel selection method used to
            build all_muaps_dist (stored in the output only). Defaults to "max_abs".
        thr_vals (np.ndarray, optional): Threshold values for clustering. Defaults to
            np.arange(0, 2.0001, 0.001).
        flag_plot (bool, optional): Flag to plot clustering metrics. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, Dict[str, np.ndarray]]: Optimal threshold information and
            clustering metrics. The dictionary contains the following keys:
            - "cluster_method" (str): The clustering method used.
            - "dist_metric" (str): The distance metric used.
            - "sel_chs_method" (str): The method used for selecting significant amplitude
                channels.
            - "labels" (np.ndarray): The cluster labels for each threshold value.
            - "n_clusters" (np.ndarray): The number of clusters for each threshold value.
            - "sil" (np.ndarray): The silhouette scores for each threshold value.
            - "chi" (np.ndarray): The Calinski-Harabasz scores for each threshold value.
            - "dbi" (np.ndarray): The Davies-Bouldin scores for each threshold value.
            - "w_dist_mean" (np.ndarray): The mean within-cluster distances for each
                threshold value.
            - "w_dist_std" (np.ndarray): The standard deviation of within-cluster distances
                for each threshold value.
            - "b_dist_mean" (np.ndarray): The mean between-cluster distances for each
                threshold value.
            - "b_dist_std" (np.ndarray): The standard deviation of between-cluster distances
                for each threshold value.
    """

    # Initialise variables
    n_units = all_muaps_dist.shape[0]

    # Transform distance from redundant form into condensed form
    all_muaps_dist_cond = spatial.distance.squareform(all_muaps_dist)

    # Build linkage between
    link = cluster.hierarchy.linkage(all_muaps_dist_cond, method=cluster_method)

    # Initialise outputs
    cluster_out = {
        "cluster_method": cluster_method,
        "dist_metric": dist_metric,
        "sel_chs_method": sel_chs_method,
        "labels": np.empty((len(thr_vals), n_units)),
        "n_clusters": np.empty(len(thr_vals)),
        "sil": np.empty(len(thr_vals)),
        "chi": np.empty(len(thr_vals)),
        "dbi": np.empty(len(thr_vals)),
        "w_dist_mean": np.empty(len(thr_vals)),
        "w_dist_std": np.empty(len(thr_vals)),
        "b_dist_mean": np.empty(len(thr_vals)),
        "b_dist_std": np.empty(len(thr_vals)),
    }

    for i, thr in enumerate(thr_vals):

        # Apply hierarchical/agglomerative clustering
        cluster_out["labels"][i] = cluster.hierarchy.fcluster(
            link, t=thr, criterion="distance"
        )

        # Number of clusters
        cluster_out["n_clusters"][i] = np.amax(cluster_out["labels"][i])

        # Clustering metrics
        if (cluster_out["n_clusters"][i] > 1) & (
            cluster_out["n_clusters"][i] < n_units
        ):
            cluster_out["sil"][i] = metrics.silhouette_score(
                all_muaps_dist, cluster_out["labels"][i], metric="precomputed"
            )
            cluster_out["chi"][i] = metrics.calinski_harabasz_score(
                all_muaps_dist, cluster_out["labels"][i]
            )
            cluster_out["dbi"][i] = metrics.davies_bouldin_score(
                all_muaps_dist, cluster_out["labels"][i]
            )
        else:
            cluster_out["sil"][i] = np.nan
            cluster_out["chi"][i] = np.nan
            cluster_out["dbi"][i] = np.nan

        # Between and within cluster distance stats
        w_dist_mean = []
        w_dist_std = []
        b_dist_mean = []
        b_dist_std = []
        np.fill_diagonal(all_muaps_dist, np.nan)

        for curr_cluster in np.unique(cluster_out["labels"][i]):
            curr_cluster_mask = cluster_out["labels"][i] == curr_cluster

            w_dist_mean.append(
                np.nanmean(all_muaps_dist[np.ix_(curr_cluster_mask, curr_cluster_mask)])
            )
            w_dist_std.append(
                np.nanstd(all_muaps_dist[np.ix_(curr_cluster_mask, curr_cluster_mask)])
            )
            b_dist_mean.append(
                np.nanmean(
                    all_muaps_dist[
                        np.ix_(curr_cluster_mask, np.logical_not(curr_cluster_mask))
                    ]
                )
            )
            b_dist_std.append(
                np.nanstd(
                    all_muaps_dist[
                        np.ix_(curr_cluster_mask, np.logical_not(curr_cluster_mask))
                    ]
                )
            )

        cluster_out["w_dist_mean"][i] = np.nanmean(w_dist_mean)
        cluster_out["w_dist_std"][i] = np.nanmean(w_dist_std)
        cluster_out["b_dist_mean"][i] = np.nanmean(b_dist_mean)
        cluster_out["b_dist_std"][i] = np.nanmean(b_dist_std)

        np.fill_diagonal(all_muaps_dist, 0)

    # Find maximum sil
    opt_sil_idx = np.nanargmax(cluster_out["sil"])
    opt_thr = thr_vals[opt_sil_idx]
    opt_sil = cluster_out["sil"][opt_sil_idx]
    opt_n_clusters = cluster_out["n_clusters"][opt_sil_idx]

    # Store output in a dataframe
    opt_thr_info = pd.DataFrame(
        {
            "cluster_method": [cluster_method],
            "dist_metric": [dist_metric],
            "sel_chs_method": [sel_chs_method],
            "opt_sil": [opt_sil],
            "opt_idx": [opt_sil_idx],
            "opt_thr": [opt_thr],
            "opt_n_clusters": [opt_n_clusters],
        }
    )

    if flag_plot:
        # Plot metrics
        _, axs = plt.subplots(6, 1, figsize=(10, 10))
        axs = np.ravel(axs)

        cluster.hierarchy.dendrogram(
            link.tolist(),
            ax=axs[0],
            color_threshold=opt_thr_info["opt_thr"].values[0],
            leaf_font_size=12,
        )
        axs[0].set_title(
            f"{cluster_method}" +
            f"max sil = {opt_thr_info['opt_sil'].values[0]:.3f}, " +
            f"thr = {opt_thr_info['opt_thr'].values[0]:.3f}"
        )
        axs[0].set_xlabel("Motor units")
        axs[0].set_ylabel("Linkage distance")

        axs[1].plot(thr_vals, cluster_out["n_clusters"])
        axs[1].axvline(opt_thr_info["opt_thr"].values[0], color="r")
        axs[1].grid(True)
        axs[1].set_xlim([thr_vals[0], thr_vals[-1]])
        axs[1].set_xlabel("Distance thresholds")
        axs[1].set_ylabel("Number of clusters")

        axs[2].plot(thr_vals, cluster_out["w_dist_mean"], "-", label="w_dist")
        axs[2].fill_between(
            thr_vals,
            cluster_out["w_dist_mean"] - cluster_out["w_dist_std"],
            cluster_out["w_dist_mean"] + cluster_out["w_dist_std"],
            alpha=0.2,
            label=None,
        )
        axs[2].plot(thr_vals, cluster_out["b_dist_mean"], "-", label="b_dist")
        axs[2].fill_between(
            thr_vals,
            cluster_out["b_dist_mean"] - cluster_out["b_dist_std"],
            cluster_out["b_dist_mean"] + cluster_out["b_dist_std"],
            alpha=0.2,
            label=None,
        )
        axs[2].axvline(opt_thr_info["opt_thr"].values[0], color="r", label="opt_thr")
        axs[2].legend(bbox_to_anchor=(1.1, 1))
        axs[2].grid(True)
        axs[2].set_xlim([thr_vals[0], thr_vals[-1]])
        axs[2].set_xlabel("Distance thresholds")
        axs[2].set_ylabel("Distance")

        axs[3].plot(thr_vals, cluster_out["sil"])
        axs[3].plot(
            opt_thr_info["opt_thr"].values[0],
            opt_thr_info["opt_sil"].values[0],
            "ro",
            markersize=6,
        )
        axs[3].axvline(opt_thr_info["opt_thr"].values[0], color="r")
        axs[3].grid(True)
        axs[3].set_xlim([thr_vals[0], thr_vals[-1]])
        axs[3].set_xlabel("Distance thresholds")
        axs[3].set_ylabel("silhouette\n(maximise)")

        axs[4].plot(thr_vals, cluster_out["dbi"])
        axs[4].axvline(opt_thr_info["opt_thr"].values[0], color="r")
        axs[4].grid(True)
        axs[4].set_xlim([thr_vals[0], thr_vals[-1]])
        axs[4].set_xlabel("Distance thresholds")
        axs[4].set_ylabel("Davies-Bouldin\n(minimise)")

        axs[5].plot(thr_vals, cluster_out["chi"])
        axs[5].axvline(opt_thr_info["opt_thr"].values[0], color="r")
        axs[5].grid(True)
        axs[5].set_xlim([thr_vals[0], thr_vals[-1]])
        axs[5].set_xlabel("Distance thresholds")
        axs[5].set_ylabel("Calinski-Harabasz\n(maximise)")

        plt.tight_layout()
        plt.show()

    return opt_thr_info, cluster_out


def pairwise(iterable: Iterable) -> Iterable[Tuple]:
    """
    pairwise(iterable) --> AB BC CD DE EF FG

    Returns an iterable of tuples containing adjacent pairs of elements from the input iterable.

    Args:
        iterable: The input iterable.

    Yields:
        A tuple containing adjacent pairs of elements from the input iterable.

    Example:
        >>> list(pairwise('ABCDEFG'))
        [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G')]
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def mark_groups(
    rows: List[int],
    cols: List[int],
    idx_i: List[int],
    idx_j: List[int],
    curr_dist_mat: np.ndarray,
    trial_labels: List[int],
    out_dict: Dict,
    graph: nx.DiGraph
) -> Tuple[Dict, nx.DiGraph]:
    """
    Mark groups based on the given rows and columns.

    Args:
        rows (List[int]): The list of row indices.
        cols (List[int]): The list of column indices.
        idx_i (List[int]): The list of unit indices for rows.
        idx_j (List[int]): The list of unit indices for columns.
        curr_dist_mat (np.ndarray): The current distance matrix.
        trial_labels (List[int]): The list of trial labels.
        out_dict (Dict): The output dictionary containing the following keys:
            - "unit1" (List[int]): The list of unit indices from the first group.
            - "unit2" (List[int]): The list of unit indices from the second group.
            - "label1" (List[int]): The list of trial labels for the units in the
                first group.
            - "label2" (List[int]): The list of trial labels for the units in the
                second group.
            - "link_dist" (List[float]): The list of linkage distances between the
                units.
            - "no_link_dist" (List[List[float]]): The list of distances between
                each unit and all other units in the same group.
        graph (nx.DiGraph): The directed graph.

    Returns:
        Tuple[Dict, nx.DiGraph]: The updated output dictionary and graph.
    """

    if len(rows) < 1 or len(cols) < 1:
        return out_dict, graph

    for row, col in zip(rows, cols):
        #  Get dist_mat values
        row_list, col_list = list(range(len(idx_i))), list(range(len(idx_j)))
        row_list.remove(row)
        col_list.remove(col)
        no_link_dist = np.concatenate(
            (curr_dist_mat[row_list, col], curr_dist_mat[row, col_list])
        )

        #  Update output
        out_dict['unit1'].append(idx_i[row])
        out_dict['unit2'].append(idx_j[col])
        out_dict['label1'].append(trial_labels[idx_i[row]])
        out_dict['label2'].append(trial_labels[idx_j[col]])
        out_dict['link_dist'].append(curr_dist_mat[row, col])
        out_dict['no_link_dist'].append(no_link_dist.tolist())

        # Update graph
        graph.add_edge(idx_i[row], idx_j[col], weight=curr_dist_mat[row, col])

    return out_dict, graph


def generate_group_sets(graph: nx.DiGraph) -> List[List[int]]:
    """
    Generate group sets based on the given directed graph.

    Args:
        graph (nx.DiGraph): A directed graph representing the connections between
            units.

    Returns:
        (List): A list of group sets, where each group set is a list of unit indices.

    Example:
        >>> graph = nx.DiGraph()
        >>> graph.add_edge(1, 2)
        >>> graph.add_edge(2, 3)
        >>> graph.add_edge(3, 4)
        >>> generate_group_sets(graph)
        [[1, 2, 3, 4]]
    """

    # Get all possible sources
    possible_sources = [node for node, in_degree in graph.in_degree() if in_degree == 0]

    #  Initialise group sets
    group_sets = []
    for source in possible_sources:
        #  Get successors
        curr_successor = list(graph.successors(source))

        #  If there are no successors, just add the source
        if len(curr_successor) < 1:
            group_sets.append([source])
            continue

        #  While the are successors add them to the path
        path = [source, curr_successor[0]]
        while list(graph.successors(curr_successor[0])):
            curr_successor = list(graph.successors(curr_successor[0]))
            path.append(curr_successor[0])
        #  Append path to set
        group_sets.append(path)

    # Sort by size
    group_counts = list(map(len, group_sets))
    idxs = np.argsort(group_counts)[::-1]

    return [group_sets[i] for i in idxs]


def generate_group_labels(group_sets: List[List[int]]) -> List[int]:
    """
    Generate group labels based on the given group sets.

    Args:
        group_sets (List): A list of group sets, where each group set is a list of
            unit indices.

    Returns:
        (List): A list of group labels, where each label corresponds to a unit index.

    Example:
        >>> generate_group_labels([[1, 2, 3, 4]])
        [1, 1, 1, 1]
    """
    units = len(np.concatenate(group_sets))
    groups = np.zeros(units).astype(int)
    for i, path in enumerate(group_sets):
        groups[path] = i + 1
    tracker = np.amax(groups) + 1
    idxs = np.nonzero(groups == 0)[0]
    for i, idx in enumerate(idxs):
        groups[idx] = tracker + i
    return groups


def apply_hungarian_algorithm(
    curr_dist_mat: np.ndarray,
    dist_thr: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Hungarian algorithm for association based on the given distance matrix.

    Args:
        curr_dist_mat (np.ndarray): The distance matrix.
        dist_thr (float): The distance threshold.

    Returns:
        (Tuple): containing the rows and columns that meet the distance threshold.

    Example:
        >>> apply_hungarian_algorithm(curr_dist_mat, dist_thr)
        (array([0, 1, 2]), array([0, 1, 2]))
    """
    rows, cols = optimize.linear_sum_assignment(curr_dist_mat)

    # Keep only the rows and cols that meet the threshold
    valid_idxs = curr_dist_mat[rows, cols] <= dist_thr
    rows, cols = rows[valid_idxs], cols[valid_idxs]
    return rows, cols


def apply_sorted_grid_search(
        curr_dist_mat: np.ndarray,
        dist_thr: float,
) -> Tuple[List[int], List[int]]:
    """
    Apply a sorted grid search to find possible matches for each row in the
    distance matrix.

    Args:
        curr_dist_mat (np.ndarray): The distance matrix.
        dist_thr (float): The distance threshold.

    Returns:
        (Tuple): containing the rows and columns that meet the distance threshold.

    Example:
        >>> apply_sorted_grid_search(curr_dist_mat, dist_thr)
        ([0, 1, 2], [0, 1, 2])
    """

    #  Search for possible matches for each row
    rows, cols = [], []
    for i in range(curr_dist_mat.shape[0]):

        # Get the minimum distance for that row
        min_dist_i = np.min(curr_dist_mat[i, :])
        if min_dist_i > dist_thr:
            continue

        # If it is a valid hit, make sure it is also the best assignment for the column
        j = np.argmin(curr_dist_mat[i, :])
        if min_dist_i > np.min(curr_dist_mat[:, j]):
            continue

        #  If it is, then mark it as a match
        rows.append(i)
        cols.append(j)

    return rows, cols


def assign_muaps_across_seq_trials(
    dist_mat: np.ndarray,
    trial_labels: np.ndarray,
    trial_set: List[int],
    assign_method: Literal["hungarian", "grid-search"] = "hungarian",
    dist_thr: Optional[float] = 0.3
) -> Tuple[Dict[str, List], nx.DiGraph, np.ndarray]:
    """
    Assign MUAPs across sequential trials based on the given distance matrix.

    Args:
        dist_mat (np.ndarray): The distance matrix.
        trial_labels (np.ndarray): All trial labels.
        trial_set (List[int]): The trial labels present in the distance matrix.
        assign_method (str, optional): The method to use for assignment:
            "hungarian" or "grid-search". Defaults to "hungarian".
        dist_thr (float, optional): The distance threshold. Defaults to 0.3.

    Returns:
        Tuple[Dict[str, List], nx.DiGraph, np.ndarray]: A tuple containing the
            output dictionary, the graph, and the updated distance matrix.
    """

    #  Initialise outputs
    keys = ["unit1", "unit2", "label1", "label2", "group", "link_dist", "no_link_dist"]
    out_dict = edict({key: [] for key in keys})

    # Generate nodes
    subset_sizes = [np.sum(trial_labels == trial) for trial in trial_set]
    extents = nx.utils.pairwise(itertools.accumulate([0,] + subset_sizes))
    layers = [range(start, end) for start, end in extents]

    #  Generate graph
    graph = nx.DiGraph()
    for trial, layer in zip(trial_set, layers):
        graph.add_nodes_from(layer, layer=trial)

    # Find adjacent neighbours
    # ------------------------
    for trial_i, trial_j in pairwise(trial_set):

        # Get indexes of the trials
        idx_i = np.nonzero(trial_labels == trial_i)[0]
        idx_j = np.nonzero(trial_labels == trial_j)[0]

        #  If there are no units in any of the trials, skip
        if len(idx_i) < 1 or len(idx_j) < 1:
            continue

        #  Assign units
        curr_dist_mat = dist_mat[np.ix_(idx_i, idx_j)]
        if assign_method == "hungarian":
            rows, cols = apply_hungarian_algorithm(curr_dist_mat, dist_thr)
        elif assign_method == "grid-search":
            rows, cols = apply_sorted_grid_search(curr_dist_mat, dist_thr)

        #  If there are no valid matches, all units are unique
        if len(rows) < 1:
            continue

        #  Mark the groups
        out_dict, graph = mark_groups(
            rows, cols, idx_i, idx_j, curr_dist_mat, trial_labels, out_dict, graph
        )

        # Once a trial has been explored, mask it
        dist_mat[np.ix_(idx_i, idx_j)] = 2

    return out_dict, graph, dist_mat


def assign_muaps_all_trials(
    muaps: np.ndarray,
    trial_labels: np.ndarray,
    trial_set: List[int],
    assign_method: Literal["hungarian", "grid-search"] = "hungarian",
    dist_metric: Literal["corr", "cosine", "nmse", "nfd"] = "nmse",
    dist_thr: float = 0.3,
    sel_chs_method: Literal["iqr", "iqr_ptp", "max_abs", "ptp"] = "iqr",
) -> Tuple[List[int], List[List[int]], pd.DataFrame, nx.DiGraph]:
    """
    Assign MUAPs across all trials based on the given distance matrix.

    Args:
        muaps (np.ndarray): MUAPs with shape (units, ch_rows, ch_cols, samples).
        trial_labels (np.ndarray): All trial labels.
        trial_set (List[int]): The trial labels of the MUAPs.
        assign_method (str, optional): The method to use for assignment:
            "hungarian" or "grid-search". Defaults to "hungarian".
        dist_metric (str, optional): The distance metric: "corr", "cosine", "nmse" or "nfd".
            Defaults to "nmse".
        dist_thr (float, optional): The distance threshold. Defaults to 0.3.
        sel_chs_method (str, optional): The method to use for selecting
            channels: "iqr", "iqr_ptp", "max_abs" or "ptp". Defaults to "iqr".

    Returns:
        Tuple[List[int], List[List[int]], pd.DataFrame, nx.DiGraph]: A tuple
            containing the group labels, group sets, link information dataframe,
            and the graph.
    """

    #  Compute distances between muaps
    #  -------------------------------
    dist_mat, _ = compute_all_muaps_dist(
        muaps, sel_chs_method=sel_chs_method, dist_metric=dist_metric
    )

    # Discard connections between the same trial
    idx_tril = np.tril_indices(dist_mat.shape[0])
    dist_mat[idx_tril] = 2
    for trial in np.unique(trial_labels):
        trial_idxs = np.nonzero(trial_labels == trial)[0]
        dist_mat[np.ix_(trial_idxs, trial_idxs)] = 2

    # fig, axs = plt.subplots(1,3, figsize=(12,4), layout='constrained', dpi=200)
    # g = sns.heatmap(dist_mat, annot=False, ax=axs[0])
    # g.set(title='Initial')

    out_dict, graph, dist_mat = assign_muaps_across_seq_trials(
        dist_mat, trial_labels, trial_set, assign_method, dist_thr
    )

    # g = sns.heatmap(dist_mat, annot=False, ax=axs[1])
    # g.set(title='After adjacent')

    # Find distant neighbours
    # -----------------------
    #  Find distant hits
    rows, cols = np.nonzero(dist_mat <= dist_thr)
    distant_dists = dist_mat[rows, cols]

    #  Sort based on distant metric
    sort_idxs = np.argsort(distant_dists)
    rows, cols = rows[sort_idxs], cols[sort_idxs]

    # Get the unique trial combinations to perform the hungarian algorithm
    distant_trials = [
        (trial_labels[row], trial_labels[col]) for row, col in zip(rows, cols)
    ]
    trial_comb = list(dict.fromkeys(distant_trials))

    #  Apply hungarian algorithm
    for trial_i, trial_j in trial_comb:

        # Get indexes of the trials
        idx_i = np.nonzero(trial_labels == trial_i)[0]
        idx_j = np.nonzero(trial_labels == trial_j)[0]

        #  Select source units within trial_i (no output edges)
        idx_i = [idx for idx in idx_i if graph.out_degree(idx) == 0]
        #  Select target units within trial_j (no input edges)
        idx_j = [idx for idx in idx_j if graph.in_degree(idx) == 0]

        if len(idx_i) < 1 or len(idx_j) < 1:
            # Once a trial has been explored, mask it
            dist_mat[np.ix_(idx_i, idx_j)] = 2
            continue

        #  Assign units
        curr_dist_mat = dist_mat[np.ix_(idx_i, idx_j)]
        if assign_method == "hungarian":
            rows, cols = apply_hungarian_algorithm(curr_dist_mat, dist_thr)
        elif assign_method == "grid-search":
            rows, cols = apply_sorted_grid_search(curr_dist_mat, dist_thr)

        #  If there are no valid matches, all units are unique
        if len(rows) < 1:
            continue

        #  Mark the groups
        out_dict, graph = mark_groups(
            rows, cols, idx_i, idx_j, curr_dist_mat, trial_labels, out_dict, graph
        )

        # Once a trial has been explored, mask it
        dist_mat[np.ix_(idx_i, idx_j)] = 2

    # g = sns.heatmap(dist_mat, annot=False, ax=axs[2])
    # g.set(title='After distant')
    # plt.show()

    #  Generate group sets and labels
    group_sets = generate_group_sets(graph)
    group_labels = generate_group_labels(group_sets)

    #  Fill groups in out_dict
    for unit in out_dict.unit1:
        out_dict.group.append(group_labels[unit])
    df_link_info = pd.DataFrame(out_dict)

    return group_labels, group_sets, df_link_info, graph
