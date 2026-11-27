"""Functions to compute MUAP similarity and distance metrics"""

import itertools
import warnings
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from scipy import cluster, optimize, signal, spatial, stats
from sklearn import metrics

ChSelMethod = Literal["iqr", "perc", "n"]
ChSelMetric = Literal["abs", "ptp"]
DistMetric = Literal["corr", "cosine", "nmse", "nfd"]
AssignMethod = Literal["hungarian", "grid-search"]

# Per-channel amplitude used to select channels
_CH_SEL_METRICS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "abs": lambda muap: np.amax(np.abs(muap), axis=-1),
    "ptp": lambda muap: np.ptp(muap, axis=-1),
}


def _get_option(options: Dict[str, Callable], name: str, arg: str) -> Callable:
    """Return options[name], or raise a ValueError listing the valid names."""
    if name not in options:
        raise ValueError(f"Unknown {arg} {name!r}; expected one of {list(options)}.")
    return options[name]


def get_sel_chs(
    muap: np.ndarray,
    ch_sel_method: Optional[ChSelMethod] = "iqr",
    ch_sel_metric: ChSelMetric = "abs",
    ch_sel_param: Optional[float] = None,
) -> np.ndarray:
    """Select the highest amplitude channels of a MUAP.

    Args:
        muap (np.ndarray): Motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        ch_sel_method (str, optional): How channels are selected from their
            amplitude: "iqr" (above the upper Tukey fence, Q3 + 1.5 IQR),
            "perc" (above the `ch_sel_param` quantile) or "n" (the
            `ch_sel_param` highest channels). None selects every channel.
            Defaults to "iqr".
        ch_sel_metric (str, optional): Channel amplitude: "abs" (peak absolute
            value) or "ptp" (peak to peak). Defaults to "abs".
        ch_sel_param (float, optional): Quantile in [0, 1] for "perc" (defaults
            to 0.95) or number of channels for "n" (defaults to 5). Must be None
            for the other methods. Defaults to None.

    Returns:
        np.ndarray: Boolean mask of the selected channels with shape
            (ch_rows, ch_cols). It is empty if no channel passes the selection.

    Raises:
        ValueError: If an option is unknown or `ch_sel_param` does not match
            `ch_sel_method`.
    """
    if ch_sel_method not in (*get_args(ChSelMethod), None):
        raise ValueError(
            f"Unknown ch_sel_method {ch_sel_method!r}; "
            f"expected one of {list(get_args(ChSelMethod))} or None."
        )
    if ch_sel_param is not None and ch_sel_method not in ("perc", "n"):
        raise ValueError(f"ch_sel_param is only used by 'perc' and 'n', not {ch_sel_method!r}.")
    amps = _get_option(_CH_SEL_METRICS, ch_sel_metric, "ch_sel_metric")(muap)

    if ch_sel_method is None:
        return np.ones(amps.shape, dtype=bool)
    if ch_sel_method == "iqr":
        q1, q3 = np.percentile(amps, [25, 75])
        return amps > q3 + 1.5 * (q3 - q1)
    if ch_sel_method == "perc":
        return amps > np.quantile(amps, 0.95 if ch_sel_param is None else ch_sel_param)

    # "n": highest n channels
    n_chs = 5 if ch_sel_param is None else int(ch_sel_param)
    ch_mask = np.zeros(amps.size, dtype=bool)
    ch_mask[np.argsort(-amps, axis=None, kind="stable")[:n_chs]] = True
    return ch_mask.reshape(amps.shape)


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

    # Compute norms (Frobenius, i.e. L2 of all samples)
    muap1_l2 = np.linalg.norm(muap1)
    muap2_l2 = np.linalg.norm(muap2)

    # Normalise muaps
    muap1_norm = (muap1 - muap1_mean[..., None]) / muap1_l2
    muap2_norm = (muap2 - muap2_mean[..., None]) / muap2_l2

    # Compute energy
    muap1_energy = np.sum(np.power(muap1_norm, 2))
    muap2_energy = np.sum(np.power(muap2_norm, 2))

    # Compute normalized farina distance
    muaps_nfd = (2 * np.sum(np.power(muap1_norm - muap2_norm, 2))) / (
        muap1_energy + muap2_energy
    )

    return muaps_nfd


# Distance between two flattened MUAPs
_DIST_FUNS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "corr": lambda muap1, muap2: 1 - stats.pearsonr(muap1, muap2)[0],
    "cosine": spatial.distance.cosine,
    "nmse": nmse,
    "nfd": norm_farina_distance,
}


def get_alignment(
        muap1: np.ndarray,
        muap2: np.ndarray,
        flag_debug: Optional[bool] = False
        ) -> int:
    """Compute the alignment lag between two MUAPs.

    The lag maximises the correlation, so inverted MUAPs are not aligned.

    Args:
        muap1 (np.ndarray): First motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        muap2 (np.ndarray): Second motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        flag_debug (bool, optional): Flag to enable debug mode. Defaults to
            False.

    Returns:
        int: The lag to apply to `muap2` (with `np.roll`) to align it to `muap1`.
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
    max_idx = np.argmax(bounded_corr)
    muaps_lag = int(bounded_lags[max_idx])

    if flag_debug:
        _, axs = plt.subplots(3, 1)

        axs[0].plot(bounded_lags, bounded_corr)
        axs[0].plot(muaps_lag, bounded_corr[max_idx], "ro")
        axs[0].set_ylabel("Correlation")
        axs[0].set_xlabel("Lags")
        axs[0].grid(True)
        axs[0].set_xlim([bounded_lags[0], bounded_lags[-1]])
        axs[0].set_title(f"max_corr = {bounded_corr[max_idx]:.3f} at lag = {muaps_lag}")

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


def compute_muaps_dist(
        muap1: np.ndarray,
        muap2: np.ndarray,
        dist_metric: DistMetric = "nmse",
        ch_sel_method: Optional[ChSelMethod] = "iqr",
        ch_sel_metric: ChSelMetric = "abs",
        ch_sel_param: Optional[float] = None,
        ) -> Tuple[float, int]:
    """Compute the distance between two MUAPs.

    The MUAPs are compared on the channels selected in either of them (all
    channels if none is selected), after aligning `muap2` to `muap1`.

    Args:
        muap1 (np.ndarray): First motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        muap2 (np.ndarray): Second motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        dist_metric (str, optional): Distance metric: "corr" (1 - Pearson
            correlation), "cosine", "nmse" or "nfd" (normalised Farina
            distance). Defaults to "nmse".
        ch_sel_method (str, optional): Channel selection method, see
            `get_sel_chs`. Defaults to "iqr".
        ch_sel_metric (str, optional): Channel amplitude for the selection,
            see `get_sel_chs`. Defaults to "abs".
        ch_sel_param (float, optional): Parameter of the selection method,
            see `get_sel_chs`. Defaults to None.

    Returns:
        Tuple[float, int]: The distance between the two MUAPs and the
            alignment lag.

    Raises:
        ValueError: If an option is unknown.
    """
    dist_fun = _get_option(_DIST_FUNS, dist_metric, "dist_metric")

    # Channels selected in either muap, or all channels if none
    ch_sel = (ch_sel_method, ch_sel_metric, ch_sel_param)
    sel_chs = np.logical_or(get_sel_chs(muap1, *ch_sel), get_sel_chs(muap2, *ch_sel))
    if not np.any(sel_chs):
        sel_chs[:] = True
    muap1_sel = muap1[sel_chs]
    muap2_sel = muap2[sel_chs]

    # Align signals based on the selected channels
    muaps_lag = get_alignment(muap1_sel, muap2_sel)
    muap2_sel = np.roll(muap2_sel, muaps_lag, axis=-1)

    return dist_fun(muap1_sel.flatten(), muap2_sel.flatten()), muaps_lag


def compute_muaps_similarity(
    muap1: np.ndarray,
    muap2: np.ndarray,
    dist_metric: DistMetric = "nmse",
    ch_sel_method: Optional[ChSelMethod] = "iqr",
    ch_sel_metric: ChSelMetric = "abs",
    ch_sel_param: Optional[float] = None,
) -> Tuple[float, int]:
    """Compute the similarity between two MUAPs, as 1 - `compute_muaps_dist`.

    Args:
        muap1 (np.ndarray): First motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        muap2 (np.ndarray): Second motor unit action potential with shape
            (ch_rows, ch_cols, samples).
        dist_metric (str, optional): Distance metric, see
            `compute_muaps_dist`. Defaults to "nmse".
        ch_sel_method (str, optional): Channel selection method, see
            `get_sel_chs`. Defaults to "iqr".
        ch_sel_metric (str, optional): Channel amplitude for the selection,
            see `get_sel_chs`. Defaults to "abs".
        ch_sel_param (float, optional): Parameter of the selection method,
            see `get_sel_chs`. Defaults to None.

    Returns:
        Tuple[float, int]: The similarity between the two MUAPs and the
            alignment lag.
    """
    muaps_dist, muaps_lag = compute_muaps_dist(
        muap1, muap2, dist_metric, ch_sel_method, ch_sel_metric, ch_sel_param
    )
    return 1 - muaps_dist, muaps_lag


def compute_muaps_dist_sets(
    muaps1: np.ndarray,
    muaps2: np.ndarray,
    dist_metric: DistMetric = "nmse",
    ch_sel_method: Optional[ChSelMethod] = "iqr",
    ch_sel_metric: ChSelMetric = "abs",
    ch_sel_param: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the distance between sets of MUAPs.

    Args:
        muaps1 (np.ndarray): First set of motor unit action potentials with shape
            (n_units1, ch_rows, ch_cols, samples).
        muaps2 (np.ndarray): Second set of motor unit action potentials with shape
            (n_units2, ch_rows, ch_cols, samples).
        dist_metric (str, optional): Distance metric, see
            `compute_muaps_dist`. Defaults to "nmse".
        ch_sel_method (str, optional): Channel selection method, see
            `get_sel_chs`. Defaults to "iqr".
        ch_sel_metric (str, optional): Channel amplitude for the selection,
            see `get_sel_chs`. Defaults to "abs".
        ch_sel_param (float, optional): Parameter of the selection method,
            see `get_sel_chs`. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The distance matrix between the two sets of
            MUAPs and the alignment lag matrix, with shape (n_units1, n_units2).
            The lag aligns each unit of `muaps2` to each unit of `muaps1`.
    """

    # Check muaps size
    if len(muaps1) == 0 or len(muaps2) == 0:
        # No units
        return np.empty((0, 0)), np.empty((0, 0), dtype=int)

    # Check units sizes
    if len(muaps1.shape) < 4:
        muaps1 = np.expand_dims(muaps1, axis=0)
    if len(muaps2.shape) < 4:
        muaps2 = np.expand_dims(muaps2, axis=0)

    # Initialise output
    n_units1 = muaps1.shape[0]
    n_units2 = muaps2.shape[0]
    all_muaps_dist = np.zeros((n_units1, n_units2))
    all_muaps_lags = np.zeros((n_units1, n_units2), dtype=int)

    # Fill distance matrix for each pair comparison
    for unit1, unit2 in itertools.product(range(n_units1), range(n_units2)):
        all_muaps_dist[unit1, unit2], all_muaps_lags[unit1, unit2] = compute_muaps_dist(
            muaps1[unit1], muaps2[unit2], dist_metric, ch_sel_method, ch_sel_metric,
            ch_sel_param,
        )

    return all_muaps_dist, all_muaps_lags


def compute_all_muaps_dist(
    muaps: np.ndarray,
    dist_metric: DistMetric = "nmse",
    ch_sel_method: Optional[ChSelMethod] = "iqr",
    ch_sel_metric: ChSelMetric = "abs",
    ch_sel_param: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the distance between all pairs of MUAPs.

    Args:
        muaps (np.ndarray): Motor unit action potentials with shape
            (n_units, ch_rows, ch_cols, samples).
        dist_metric (str, optional): Distance metric, see
            `compute_muaps_dist`. Defaults to "nmse".
        ch_sel_method (str, optional): Channel selection method, see
            `get_sel_chs`. Defaults to "iqr".
        ch_sel_metric (str, optional): Channel amplitude for the selection,
            see `get_sel_chs`. Defaults to "abs".
        ch_sel_param (float, optional): Parameter of the selection method,
            see `get_sel_chs`. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The symmetric distance matrix between all
            pairs of MUAPs and the antisymmetric alignment lag matrix, with shape
            (n_units, n_units). `lags[i, j]` aligns unit j to unit i.
    """

    # Check muaps size
    if len(muaps) == 0:
        # No units
        return np.empty((0, 0)), np.empty((0, 0), dtype=int)
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)

    # Initialise output
    n_units = muaps.shape[0]
    all_muaps_dist = np.zeros((n_units, n_units))
    all_muaps_lags = np.zeros((n_units, n_units), dtype=int)

    # Fill distance matrix for each unique pair comparison
    for unit1, unit2 in itertools.combinations(range(n_units), 2):
        curr_muaps_dist, curr_muaps_lag = compute_muaps_dist(
            muaps[unit1], muaps[unit2], dist_metric, ch_sel_method, ch_sel_metric,
            ch_sel_param,
        )
        all_muaps_dist[unit1, unit2] = all_muaps_dist[unit2, unit1] = curr_muaps_dist
        all_muaps_lags[unit1, unit2] = curr_muaps_lag
        all_muaps_lags[unit2, unit1] = -curr_muaps_lag

    return all_muaps_dist, all_muaps_lags


def _fill_value(dist: np.ndarray, dist_thr: float = 0) -> int:
    """Integer distance above every finite distance and the threshold, to mask pairs."""
    max_dist = np.max(dist, initial=0, where=np.isfinite(dist))
    return int(np.ceil(max(max_dist, dist_thr))) + 1


def mask_within_trial_dist(
    all_muaps_dist: np.ndarray,
    trial_labels: np.ndarray,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """Set the distance between different MUAPs of the same trial to a large value.

    Units identified in the same trial that are different motor units, should
    not be matched together. The diagonal (each unit with itself) is left at 0.

    Args:
        all_muaps_dist (np.ndarray): Distance matrix between all pairs of MUAPs
            with shape (n_units, n_units).
        trial_labels (np.ndarray): Trial label of each MUAP, with shape (n_units).
        fill_value (float, optional): Distance assigned to pairs from the same
            trial. Defaults to 1 + the largest distance in the matrix, rounded up.

    Returns:
        np.ndarray: A masked copy of the distance matrix. The input is not modified.
    """
    trial_labels = np.asarray(trial_labels)
    if trial_labels.shape != (all_muaps_dist.shape[0],):
        raise ValueError(
            f"trial_labels has shape {trial_labels.shape}, expected "
            f"({all_muaps_dist.shape[0]},) to match the distance matrix."
        )
    if fill_value is None:
        fill_value = _fill_value(all_muaps_dist)

    same_trial = trial_labels[:, None] == trial_labels[None, :]
    np.fill_diagonal(same_trial, False)

    masked_dist = np.array(all_muaps_dist, dtype=float)
    masked_dist[same_trial] = fill_value
    return masked_dist


def _nanmean_or_nan(values: np.ndarray) -> float:
    """np.nanmean, returning NaN without a RuntimeWarning when there is nothing to average."""
    values = np.asarray(values)
    return np.nanmean(values) if np.any(~np.isnan(values)) else np.nan


def _nanstd_or_nan(values: np.ndarray) -> float:
    """np.nanstd, returning NaN without a RuntimeWarning when there is nothing to average."""
    values = np.asarray(values)
    return np.nanstd(values) if np.any(~np.isnan(values)) else np.nan


def _warn_same_trial_clusters(
    labels: np.ndarray, trial_labels: np.ndarray, thr: float
) -> None:
    """Warn about clusters that contain more than one unit from the same trial."""
    conflicts = []
    for curr_cluster in np.unique(labels):
        cluster_units = np.nonzero(labels == curr_cluster)[0]
        cluster_trials = trial_labels[cluster_units]
        for trial in np.unique(cluster_trials):
            trial_units = cluster_units[cluster_trials == trial]
            if len(trial_units) > 1:
                conflicts.append(
                    f"cluster {int(curr_cluster)}: units {trial_units.tolist()} "
                    f"(trial {trial})"
                )

    if conflicts:
        warnings.warn(
            f"At the selected threshold {thr:.3f}, units from the same trial were "
            "merged into the same cluster: " + "; ".join(conflicts) + ". "
            "Use cluster_method='complete' to keep units from the same trial apart.",
            UserWarning,
            stacklevel=3,
        )


def cluster_muaps(
    all_muaps_dist: np.ndarray,
    cluster_method: Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ] = "ward",
    dist_metric: DistMetric = "nmse",
    ch_sel_method: Optional[ChSelMethod] = "iqr",
    ch_sel_metric: ChSelMetric = "abs",
    ch_sel_param: Optional[float] = None,
    thr_vals: Optional[np.ndarray] = np.arange(0, 2.0001, 0.001),  # noqa: B008 (read-only)
    flag_plot: Optional[bool] = True,
    *,
    trial_labels: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Cluster MUAPs based on distance matrix.

    Args:
        all_muaps_dist (np.ndarray): Distance matrix between all pairs of MUAPs.
            It is not modified.
        cluster_method (str, optional): Linkage method passed to
            scipy.cluster.hierarchy.linkage. Defaults to "ward".
        dist_metric (str, optional): Distance metric used to build
            all_muaps_dist (stored in the output only). Defaults to "nmse".
        ch_sel_method (str, optional): Channel selection method used to
            build all_muaps_dist (stored in the output only). Defaults to "iqr".
        ch_sel_metric (str, optional): Channel amplitude used to build
            all_muaps_dist (stored in the output only). Defaults to "abs".
        ch_sel_param (float, optional): Channel selection parameter used to
            build all_muaps_dist (stored in the output only). Defaults to None.
        thr_vals (np.ndarray, optional): Threshold values for clustering. Defaults to
            np.arange(0, 2.0001, 0.001).
        flag_plot (bool, optional): Flag to plot clustering metrics. Defaults to True.
        trial_labels (np.ndarray, optional): Trial label of each MUAP, with shape
            (n_units). If given, MUAPs from the same trial are made maximally distant
            before clustering (see `mask_within_trial_dist`), so they are not grouped
            together. Only "complete" linkage guarantees this; with other methods a
            warning lists any same-trial units merged at the selected threshold.
            The clustering metrics are always computed on the original distances.
            Defaults to None (no trial constraint).

    Returns:
        Tuple[pd.DataFrame, Dict[str, np.ndarray]]: Optimal threshold information and
            clustering metrics. The dictionary contains the following keys:
            - "cluster_method" (str): The clustering method used.
            - "dist_metric" (str): The distance metric used.
            - "ch_sel_method" (str): The channel selection method used.
            - "ch_sel_metric" (str): The channel amplitude used for the selection.
            - "ch_sel_param" (float): The channel selection parameter used.
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

    Raises:
        ValueError: If no threshold in `thr_vals` gives between 2 and n_units - 1
            clusters, so the silhouette cannot select one.
    """

    # Initialise variables (on a copy, so the input is not modified)
    all_muaps_dist = np.array(all_muaps_dist, dtype=float)
    n_units = all_muaps_dist.shape[0]

    # Distances without the diagonal, for the within and between cluster stats
    pair_dist = all_muaps_dist.copy()
    np.fill_diagonal(pair_dist, np.nan)

    # Keep units from the same trial apart (only affects the linkage, not the metrics)
    link_dist = all_muaps_dist
    if trial_labels is not None:
        trial_labels = np.asarray(trial_labels)
        link_dist = mask_within_trial_dist(
            all_muaps_dist, trial_labels, fill_value=_fill_value(all_muaps_dist, np.max(thr_vals))
        )

    # Transform distance from redundant form into condensed form
    all_muaps_dist_cond = spatial.distance.squareform(link_dist)

    # Build linkage between
    link = cluster.hierarchy.linkage(all_muaps_dist_cond, method=cluster_method)

    # Initialise outputs
    cluster_out = {
        "cluster_method": cluster_method,
        "dist_metric": dist_metric,
        "ch_sel_method": ch_sel_method,
        "ch_sel_metric": ch_sel_metric,
        "ch_sel_param": ch_sel_param,
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

        for curr_cluster in np.unique(cluster_out["labels"][i]):
            curr_cluster_mask = cluster_out["labels"][i] == curr_cluster

            # Single-unit clusters have no within distances and a single cluster has
            # no between distances: these are NaN (skipped quietly, not warned about)
            within_dists = pair_dist[np.ix_(curr_cluster_mask, curr_cluster_mask)]
            between_dists = pair_dist[
                np.ix_(curr_cluster_mask, np.logical_not(curr_cluster_mask))
            ]
            w_dist_mean.append(_nanmean_or_nan(within_dists))
            w_dist_std.append(_nanstd_or_nan(within_dists))
            b_dist_mean.append(_nanmean_or_nan(between_dists))
            b_dist_std.append(_nanstd_or_nan(between_dists))

        cluster_out["w_dist_mean"][i] = _nanmean_or_nan(w_dist_mean)
        cluster_out["w_dist_std"][i] = _nanmean_or_nan(w_dist_std)
        cluster_out["b_dist_mean"][i] = _nanmean_or_nan(b_dist_mean)
        cluster_out["b_dist_std"][i] = _nanmean_or_nan(b_dist_std)

    # Find maximum sil
    if np.all(np.isnan(cluster_out["sil"])):
        raise ValueError(
            "No threshold in thr_vals gives between 2 and n_units - 1 clusters, "
            "so the silhouette cannot select one."
        )
    opt_sil_idx = np.nanargmax(cluster_out["sil"])
    opt_thr = thr_vals[opt_sil_idx]
    opt_sil = cluster_out["sil"][opt_sil_idx]
    opt_n_clusters = cluster_out["n_clusters"][opt_sil_idx]

    # Check the trial constraint held at the selected threshold
    if trial_labels is not None:
        _warn_same_trial_clusters(cluster_out["labels"][opt_sil_idx], trial_labels, opt_thr)

    # Store output in a dataframe
    opt_thr_info = pd.DataFrame(
        {
            "cluster_method": [cluster_method],
            "dist_metric": [dist_metric],
            "ch_sel_method": [ch_sel_method],
            "ch_sel_metric": [ch_sel_metric],
            "ch_sel_param": [ch_sel_param],
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
            f"{cluster_method}: " +
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
    return zip(a, b, strict=False)


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

    for row, col in zip(rows, cols, strict=True):
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
            unit indices. Unit indices must be 0-based and cover every unit
            (0 to n_units - 1), as produced by generate_group_sets.

    Returns:
        (List): A list of group labels (starting at 1), where each label
            corresponds to a unit index.

    Example:
        >>> generate_group_labels([[0, 2, 3], [1]])
        [1, 2, 1, 1]
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
        curr_dist_mat (np.ndarray): The distance matrix. NaN distances (e.g.
            from flat MUAPs) are never matched.
        dist_thr (float): The distance threshold.

    Returns:
        (Tuple): containing the rows and columns that meet the distance threshold.

    Example:
        >>> apply_hungarian_algorithm(curr_dist_mat, dist_thr)
        (array([0, 1, 2]), array([0, 1, 2]))
    """
    # linear_sum_assignment rejects NaN: replace them by a distance above the threshold
    curr_dist_mat = np.where(
        np.isnan(curr_dist_mat), _fill_value(curr_dist_mat, dist_thr), curr_dist_mat
    )
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
        curr_dist_mat (np.ndarray): The distance matrix. NaN distances (e.g.
            from flat MUAPs) are never matched.
        dist_thr (float): The distance threshold.

    Returns:
        (Tuple): containing the rows and columns that meet the distance threshold.

    Example:
        >>> apply_sorted_grid_search(curr_dist_mat, dist_thr)
        ([0, 1, 2], [0, 1, 2])
    """
    # NaN would win np.min/np.argmin: replace them by a distance above the threshold
    curr_dist_mat = np.where(
        np.isnan(curr_dist_mat), _fill_value(curr_dist_mat, dist_thr), curr_dist_mat
    )

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


_ASSIGN_FUNS: Dict[str, Callable] = {
    "hungarian": apply_hungarian_algorithm,
    "grid-search": apply_sorted_grid_search,
}


def _check_trial_order(trial_labels: np.ndarray, trial_set: List[int]) -> None:
    """Raise a ValueError unless the MUAPs are sorted by trial, in trial_set order."""
    counts = [np.sum(trial_labels == trial) for trial in trial_set]
    if not np.array_equal(trial_labels, np.repeat(trial_set, counts)):
        raise ValueError(
            "trial_labels must be sorted in trial_set order, and every label must be "
            "in trial_set."
        )


def assign_muaps_across_seq_trials(
    dist_mat: np.ndarray,
    trial_labels: np.ndarray,
    trial_set: List[int],
    assign_method: AssignMethod = "grid-search",
    dist_thr: Optional[float] = 0.2
) -> Tuple[Dict[str, List], nx.DiGraph, np.ndarray]:
    """
    Assign MUAPs across sequential trials based on the given distance matrix.

    Args:
        dist_mat (np.ndarray): The distance matrix.
        trial_labels (np.ndarray): Trial label of each MUAP, sorted in
            `trial_set` order.
        trial_set (List[int]): The trial labels present in the distance matrix.
        assign_method (str, optional): The method to use for assignment:
            "hungarian" or "grid-search". Defaults to "grid-search".
        dist_thr (float, optional): The distance threshold. Defaults to 0.2.

    Returns:
        Tuple[Dict[str, List], nx.DiGraph, np.ndarray]: A tuple containing the
            output dictionary, the graph, and the updated distance matrix.

    Raises:
        ValueError: If `assign_method` is unknown or `trial_labels` is not
            sorted in `trial_set` order.
    """
    trial_labels = np.asarray(trial_labels)
    _check_trial_order(trial_labels, trial_set)
    assign_fun = _get_option(_ASSIGN_FUNS, assign_method, "assign_method")
    fill_value = _fill_value(dist_mat, dist_thr)

    #  Initialise outputs
    keys = ["unit1", "unit2", "label1", "label2", "group", "link_dist", "no_link_dist"]
    out_dict = edict({key: [] for key in keys})

    # Generate nodes
    subset_sizes = [np.sum(trial_labels == trial) for trial in trial_set]
    extents = nx.utils.pairwise(itertools.accumulate([0,] + subset_sizes))
    layers = [range(start, end) for start, end in extents]

    #  Generate graph
    graph = nx.DiGraph()
    for trial, layer in zip(trial_set, layers, strict=True):
        graph.add_nodes_from(layer, layer=trial)

    # Find adjacent neighbours
    # ------------------------
    for trial_i, trial_j in pairwise(trial_set):

        # Get indexes of the trials
        idx_i = np.nonzero(trial_labels == trial_i)[0]
        idx_j = np.nonzero(trial_labels == trial_j)[0]

        #  If there are no units in any of the trials, skip
        if len(idx_i) < 1 or len(idx_j) < 1:
            continue

        #  Assign units
        curr_dist_mat = dist_mat[np.ix_(idx_i, idx_j)]
        rows, cols = assign_fun(curr_dist_mat, dist_thr)

        #  If there are no valid matches, all units are unique
        if len(rows) < 1:
            continue

        #  Mark the groups
        out_dict, graph = mark_groups(
            rows, cols, idx_i, idx_j, curr_dist_mat, trial_labels, out_dict, graph
        )

        # Once a trial has been explored, mask it
        dist_mat[np.ix_(idx_i, idx_j)] = fill_value

    return out_dict, graph, dist_mat


def assign_muaps_all_trials(
    muaps: np.ndarray,
    trial_labels: np.ndarray,
    trial_set: List[int],
    assign_method: AssignMethod = "hungarian",
    dist_metric: DistMetric = "nmse",
    dist_thr: float = 0.2,
    ch_sel_method: Optional[ChSelMethod] = "iqr",
    ch_sel_metric: ChSelMetric = "abs",
    ch_sel_param: Optional[float] = None,
) -> Tuple[List[int], List[List[int]], pd.DataFrame, nx.DiGraph]:
    """
    Assign MUAPs across all trials based on the given distance matrix.

    Args:
        muaps (np.ndarray): MUAPs with shape (units, ch_rows, ch_cols, samples).
        trial_labels (np.ndarray): Trial label of each MUAP. The MUAPs must be
            sorted by trial, in `trial_set` order.
        trial_set (List[int]): The trial labels of the MUAPs, in the order
            used to match neighbouring trials.
        assign_method (str, optional): The method to use for assignment:
            "hungarian" or "grid-search". Defaults to "hungarian".
        dist_metric (str, optional): Distance metric, see
            `compute_muaps_dist`. Defaults to "nmse".
        dist_thr (float, optional): The distance threshold. Defaults to 0.2.
        ch_sel_method (str, optional): Channel selection method, see
            `get_sel_chs`. Defaults to "iqr".
        ch_sel_metric (str, optional): Channel amplitude for the selection,
            see `get_sel_chs`. Defaults to "abs".
        ch_sel_param (float, optional): Parameter of the selection method,
            see `get_sel_chs`. Defaults to None.

    Returns:
        Tuple[List[int], List[List[int]], pd.DataFrame, nx.DiGraph]: A tuple
            containing the group labels, group sets, link information dataframe,
            and the graph.

    Raises:
        ValueError: If an option is unknown or `trial_labels` is not sorted in
            `trial_set` order.
    """
    # Check inputs before computing the distances
    trial_labels = np.asarray(trial_labels)
    _check_trial_order(trial_labels, trial_set)
    assign_fun = _get_option(_ASSIGN_FUNS, assign_method, "assign_method")

    #  Compute distances between muaps
    #  -------------------------------
    dist_mat, _ = compute_all_muaps_dist(
        muaps, dist_metric, ch_sel_method, ch_sel_metric, ch_sel_param
    )

    # Discard connections within a trial and towards earlier trials
    fill_value = _fill_value(dist_mat, dist_thr)
    dist_mat = mask_within_trial_dist(dist_mat, trial_labels, fill_value)
    dist_mat[np.tril_indices(dist_mat.shape[0])] = fill_value

    out_dict, graph, dist_mat = assign_muaps_across_seq_trials(
        dist_mat, trial_labels, trial_set, assign_method, dist_thr
    )

    # Find distant neighbours
    # -----------------------
    #  Find distant hits
    rows, cols = np.nonzero(dist_mat <= dist_thr)
    distant_dists = dist_mat[rows, cols]

    #  Sort based on distant metric
    sort_idxs = np.argsort(distant_dists)
    rows, cols = rows[sort_idxs], cols[sort_idxs]

    # Get the unique trial combinations to perform the hungarian algorithm
    distant_trials = [
        (trial_labels[row], trial_labels[col]) for row, col in zip(rows, cols, strict=True)
    ]
    trial_comb = list(dict.fromkeys(distant_trials))

    #  Apply hungarian algorithm
    for trial_i, trial_j in trial_comb:

        # Get indexes of the trials
        idx_i = np.nonzero(trial_labels == trial_i)[0]
        idx_j = np.nonzero(trial_labels == trial_j)[0]

        #  Select source units within trial_i (no output edges)
        idx_i = [idx for idx in idx_i if graph.out_degree(idx) == 0]
        #  Select target units within trial_j (no input edges)
        idx_j = [idx for idx in idx_j if graph.in_degree(idx) == 0]

        if len(idx_i) < 1 or len(idx_j) < 1:
            # Once a trial has been explored, mask it
            dist_mat[np.ix_(idx_i, idx_j)] = fill_value
            continue

        #  Assign units
        curr_dist_mat = dist_mat[np.ix_(idx_i, idx_j)]
        rows, cols = assign_fun(curr_dist_mat, dist_thr)

        #  If there are no valid matches, all units are unique
        if len(rows) < 1:
            continue

        #  Mark the groups
        out_dict, graph = mark_groups(
            rows, cols, idx_i, idx_j, curr_dist_mat, trial_labels, out_dict, graph
        )

        # Once a trial has been explored, mask it
        dist_mat[np.ix_(idx_i, idx_j)] = fill_value

    #  Generate group sets and labels
    group_sets = generate_group_sets(graph)
    group_labels = generate_group_labels(group_sets)

    #  Fill groups in out_dict
    for unit in out_dict.unit1:
        out_dict.group.append(group_labels[unit])
    df_link_info = pd.DataFrame(out_dict)

    return group_labels, group_sets, df_link_info, graph
