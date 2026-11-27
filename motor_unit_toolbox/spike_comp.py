"""Compare spike trains: paired units, one-to-one matching across sets and duplicates within a set.

Spike trains are given either as a binary array of shape (samples, units) or (samples,),
or as a list with one integer array of spike times (in samples) per unit.
"""

import warnings
from typing import List, Literal, Optional, Sequence, Tuple, Union, get_args

import numpy as np
import pandas as pd
from scipy import optimize

from motor_unit_toolbox.muap_comp import _get_option
from motor_unit_toolbox.props import _check_mu_format

SpikeTrains = Union[np.ndarray, Sequence[np.ndarray]]
AgreementMetric = Literal["roa", "f1", "precision", "sensitivity", "match"]
SymmetricMetric = Literal["roa", "f1", "match"]  # same score either way round

# Denominator of each score, as a function of (tp, n_ref, n_test)
_SCORE_DENS = {
    "roa": lambda tp, n_ref, n_test: n_ref + n_test - tp,
    "f1": lambda tp, n_ref, n_test: (n_ref + n_test) / 2,
    "precision": lambda tp, n_ref, n_test: n_test,
    "sensitivity": lambda tp, n_ref, n_test: n_ref,
    "match": lambda tp, n_ref, n_test: np.maximum(n_ref, n_test),
}
_FRAME_METRICS = ("roa", "f1", "precision", "sensitivity")


def _agreement_score(tp, n_ref, n_test, metric: AgreementMetric) -> np.ndarray:
    """Agreement score from common-spike counts, 0 where undefined.

    Args:
        tp (array-like): Common spikes (true positives).
        n_ref (array-like): Spikes in the reference trains.
        n_test (array-like): Spikes in the test trains.
        metric (AgreementMetric): "roa" TP/(TP+FN+FP), "f1" 2TP/(2TP+FN+FP),
            "precision" TP/(TP+FP), "sensitivity" TP/(TP+FN), or "match"
            TP/max(n_ref, n_test).

    Returns:
        np.ndarray: Scores in [0, 1], broadcast shape of the inputs.
    """
    den_fun = _get_option(_SCORE_DENS, metric, "metric")
    tp, n_ref, n_test = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (tp, n_ref, n_test))
    )
    den = den_fun(tp, n_ref, n_test)
    return np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)


def _to_samples(ms: float, fs: float) -> int:
    """Milliseconds to whole samples."""
    return round(ms / 1000 * fs)


def _spike_times(spike_trains: SpikeTrains) -> List[np.ndarray]:
    """Sorted spike samples of each unit.

    Args:
        spike_trains (SpikeTrains): Binary array (samples, units) or (samples,), or a
            list with one integer array of spike samples per unit.

    Returns:
        List[np.ndarray]: Sorted spike samples of each unit.

    Raises:
        ValueError: If a list element is not an integer array.
    """
    if isinstance(spike_trains, np.ndarray):
        trains = _check_mu_format(spike_trains)
        return [np.flatnonzero(trains[:, unit]) for unit in range(trains.shape[1])]
    times = [np.asarray(t) for t in spike_trains]
    if any(t.size and not np.issubdtype(t.dtype, np.integer) for t in times):
        raise ValueError("Spike times must be integer sample indices.")
    return [np.sort(t.astype(int).ravel()) for t in times]


def _n_spikes(times: List[np.ndarray]) -> np.ndarray:
    """Number of spikes of each unit."""
    return np.array([len(t) for t in times], dtype=int)


def _best_lag(t_ref: np.ndarray, t_test: np.ndarray, max_lag: int, tol: int) -> int:
    """Lag in [-max_lag, max_lag] that best aligns t_test + lag with t_ref.

    Histograms the differences of all spike pairs closer than max_lag + tol and
    smooths them with a triangle of half-width tol + 1. Ties go to the smallest |lag|.

    Args:
        t_ref (np.ndarray): Sorted reference spike samples.
        t_test (np.ndarray): Sorted test spike samples.
        max_lag (int): Largest lag searched, in samples.
        tol (int): Spike coincidence tolerance, in samples.

    Returns:
        int: The lag, in samples.
    """
    w = max_lag + tol
    lo = np.searchsorted(t_test, t_ref - w, side="left")
    hi = np.searchsorted(t_test, t_ref + w, side="right")
    n_close = hi - lo
    # Ragged arange: range(lo[i], hi[i]) for every reference spike, concatenated
    offsets = np.cumsum(n_close) - n_close
    j = np.repeat(lo - offsets, n_close) + np.arange(n_close.sum())
    diffs = np.repeat(t_ref, n_close) - t_test[j]
    hist = np.bincount(diffs + w, minlength=2 * w + 1)
    weight = np.convolve(hist, np.bartlett(2 * tol + 3)[1:-1], mode="valid")
    lags = np.flatnonzero(weight == weight.max()) - max_lag
    return int(lags[np.argmin(np.abs(lags))])


def _count_common(t_ref: np.ndarray, t_test: np.ndarray, tol: int) -> int:
    """Spikes matched one-to-one within ±tol, by a two-pointer sweep of sorted trains."""
    t_ref, t_test = t_ref.tolist(), t_test.tolist()
    i = j = common = 0
    while i < len(t_ref) and j < len(t_test):
        diff = t_ref[i] - t_test[j]
        if abs(diff) <= tol:
            common, i, j = common + 1, i + 1, j + 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return common


def _compare(t_ref: np.ndarray, t_test: np.ndarray, max_lag: int, tol: int) -> Tuple[int, int]:
    """Lag and common spikes of one pair of trains."""
    lag = _best_lag(t_ref, t_test, max_lag, tol)
    return lag, _count_common(t_ref, t_test + lag, tol)


def _prepare(
    spike_trains_ref: SpikeTrains,
    spike_trains_test: SpikeTrains,
    fs: float,
    tol_spike_ms: float,
    max_lag_ms: float,
    same_units: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """Validate two sets; return their spike times, max lag and tolerance in samples.

    Raises:
        ValueError: If two arrays have different numbers of samples, or if
            same_units and the sets have different numbers of units.
    """
    times_ref, times_test = _spike_times(spike_trains_ref), _spike_times(spike_trains_test)
    arrays = isinstance(spike_trains_ref, np.ndarray) and isinstance(spike_trains_test, np.ndarray)
    if arrays and len(spike_trains_ref) != len(spike_trains_test):
        raise ValueError(
            f"Samples mismatch between ref ({len(spike_trains_ref)}) "
            f"and test ({len(spike_trains_test)})."
        )
    if same_units and len(times_ref) != len(times_test):
        raise ValueError(
            f"Units mismatch between ref ({len(times_ref)}) and test ({len(times_test)})."
        )
    return times_ref, times_test, _to_samples(max_lag_ms, fs), _to_samples(tol_spike_ms, fs)


def _agreement_counts(
    spike_trains_ref: SpikeTrains,
    spike_trains_test: Optional[SpikeTrains],
    fs: float,
    tol_spike_ms: float,
    max_lag_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lag and common spikes of every (ref, test) unit pair, and the spikes per unit.

    A None test set compares ref with itself: only the upper triangle is computed and
    mirrored, and each unit matches itself fully.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: lag and tp, shape
        (n_ref, n_test), and the spikes of each ref and test unit.
    """
    within = spike_trains_test is None
    times_ref, times_test, max_lag, tol = _prepare(
        spike_trains_ref,
        spike_trains_ref if within else spike_trains_test,
        fs,
        tol_spike_ms,
        max_lag_ms,
        same_units=within,
    )
    lag = np.zeros((len(times_ref), len(times_test)), dtype=int)
    tp = np.zeros_like(lag)
    for r, t_ref in enumerate(times_ref):
        for t in range(r + 1 if within else 0, len(times_test)):
            lag[r, t], tp[r, t] = _compare(t_ref, times_test[t], max_lag, tol)
    n_ref, n_test = _n_spikes(times_ref), _n_spikes(times_test)
    if within:
        lag, tp = lag - lag.T, tp + tp.T + np.diag(n_ref)
    return lag, tp, n_ref, n_test


def _pairs_frame(ref_idx, test_idx, lag, tp, n_ref, n_test) -> pd.DataFrame:
    """DataFrame of unit pairs with their lag, counts and every metric."""
    tp, n_ref, n_test = (np.asarray(x, dtype=int) for x in (tp, n_ref, n_test))
    frame = pd.DataFrame(
        {
            "ref_idx": ref_idx,
            "test_idx": test_idx,
            "lag": lag,
            "tp": tp,
            "fn": n_ref - tp,
            "fp": n_test - tp,
        }
    )
    for metric in _FRAME_METRICS:
        frame[metric] = _agreement_score(tp, n_ref, n_test, metric)
    return frame


def compare_paired_spikes(
    spike_trains_ref: SpikeTrains,
    spike_trains_test: SpikeTrains,
    *,
    fs: float,
    tol_spike_ms: float = 0.5,
    max_lag_ms: float = 40.0,
) -> pd.DataFrame:
    """Agreement of paired spike trains: unit i of ref against unit i of test.

    Args:
        spike_trains_ref (SpikeTrains): Reference trains: binary array (samples, n) or
            (samples,), or a list with one integer array of spike samples per unit.
        spike_trains_test (SpikeTrains): Test trains, same format and number of units.
        fs (float): Sampling frequency in Hz.
        tol_spike_ms (float): Spike coincidence tolerance in ms. Defaults to 0.5.
        max_lag_ms (float): Largest lag searched between two trains in ms. Defaults to 40.

    Returns:
        pd.DataFrame: One row per unit with ref_idx, test_idx, lag (samples added to
        the test spikes to align them), tp, fn, fp, roa, f1, precision and
        sensitivity, taking ref as ground truth. Scores are 0 where undefined.

    Raises:
        ValueError: If the sets differ in samples (arrays) or units.
    """
    times_ref, times_test, max_lag, tol = _prepare(
        spike_trains_ref, spike_trains_test, fs, tol_spike_ms, max_lag_ms, same_units=True
    )
    lag_tp = np.array(
        [_compare(r, t, max_lag, tol) for r, t in zip(times_ref, times_test, strict=True)],
        dtype=int,
    ).reshape(-1, 2)
    units = np.arange(len(times_ref))
    return _pairs_frame(
        units, units, lag_tp[:, 0], lag_tp[:, 1], _n_spikes(times_ref), _n_spikes(times_test)
    )


def compare_spikes(
    spike_trains_ref: SpikeTrains,
    spike_trains_test: SpikeTrains,
    *,
    fs: float,
    match_metric: AgreementMetric = "roa",
    min_score: float = 0.0,
    report_unmatched: bool = False,
    tol_spike_ms: float = 0.5,
    max_lag_ms: float = 40.0,
) -> pd.DataFrame:
    """Pair the units of two sets one-to-one and compute their agreement.

    Every (ref, test) pair is aligned and scored; the Hungarian algorithm then picks
    the one-to-one pairing with the highest total match_metric. Remove duplicates
    within each set first (`find_duplicates`), as a second copy of a unit stays unpaired.

    Args:
        spike_trains_ref (SpikeTrains): Reference (ground truth) trains: binary array
            (samples, n_ref) or (samples,), or a list with one integer array of spike
            samples per unit.
        spike_trains_test (SpikeTrains): Test trains, same format.
        fs (float): Sampling frequency in Hz.
        match_metric (AgreementMetric): Score the pairing maximises: "roa", "f1",
            "precision", "sensitivity" or "match". Defaults to "roa".
        min_score (float): Pairs scoring below it on match_metric are unmatched.
            Defaults to 0.
        report_unmatched (bool): Also return a row for each unmatched test unit, with
            ref_idx and lag <NA>, all its spikes as fp, tp = fn = 0 and every score 0.
            Defaults to False.
        tol_spike_ms (float): Spike coincidence tolerance in ms. Defaults to 0.5.
        max_lag_ms (float): Largest lag searched between two trains in ms. Defaults to 40.

    Returns:
        pd.DataFrame: Columns as in `compare_paired_spikes`, ordered by test unit. One
        row per matched pair (at least one common spike and min_score), or per test
        unit with report_unmatched.

    Raises:
        ValueError: If two arrays have different numbers of samples.
    """
    lag, tp, n_ref, n_test = _agreement_counts(
        spike_trains_ref, spike_trains_test, fs, tol_spike_ms, max_lag_ms
    )
    score = _agreement_score(tp, n_ref[:, None], n_test[None, :], match_metric)
    rows, cols = optimize.linear_sum_assignment(score, maximize=True)
    keep = (tp[rows, cols] > 0) & (score[rows, cols] >= min_score)
    order = np.argsort(cols[keep])
    rows, cols = rows[keep][order], cols[keep][order]
    if report_unmatched:
        # Unmatched test units point to an extra, empty reference unit (index -1)
        lag, tp = (np.vstack([x, np.zeros((1, len(n_test)), dtype=int)]) for x in (lag, tp))
        n_ref = np.append(n_ref, 0)
        ref_of = np.full(len(n_test), -1)
        ref_of[cols] = rows
        rows, cols = ref_of, np.arange(len(n_test))
    frame = _pairs_frame(rows, cols, lag[rows, cols], tp[rows, cols], n_ref[rows], n_test[cols])
    if report_unmatched:
        frame = frame.astype({"ref_idx": "Int64", "lag": "Int64"})
        frame.loc[rows < 0, ["ref_idx", "lag"]] = pd.NA
    return frame


def compute_agreement_matrix(
    spike_trains_ref: SpikeTrains,
    spike_trains_test: Optional[SpikeTrains] = None,
    *,
    fs: float,
    metric: AgreementMetric = "roa",
    tol_spike_ms: float = 0.5,
    max_lag_ms: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Agreement score and lag of every (ref, test) unit pair.

    Args:
        spike_trains_ref (SpikeTrains): Reference trains: binary array (samples, n_ref)
            or (samples,), or a list with one integer array of spike samples per unit.
        spike_trains_test (SpikeTrains, optional): Test trains, same format. None
            compares ref with itself (diagonal: 1 for units with spikes, lag 0).
            Defaults to None.
        fs (float): Sampling frequency in Hz.
        metric (AgreementMetric): "roa", "f1", "precision", "sensitivity" or "match".
            Defaults to "roa".
        tol_spike_ms (float): Spike coincidence tolerance in ms. Defaults to 0.5.
        max_lag_ms (float): Largest lag searched between two trains in ms. Defaults to 40.

    Returns:
        Tuple[np.ndarray, np.ndarray]: score and lag (samples added to the test spikes
        to align them), each with shape (n_ref, n_test).

    Raises:
        ValueError: If two arrays have different numbers of samples.
    """
    lag, tp, n_ref, n_test = _agreement_counts(
        spike_trains_ref, spike_trains_test, fs, tol_spike_ms, max_lag_ms
    )
    return _agreement_score(tp, n_ref[:, None], n_test[None, :], metric), lag


def find_duplicates(
    spike_trains: SpikeTrains,
    quality: np.ndarray,
    *,
    fs: float,
    metric: SymmetricMetric = "roa",
    threshold: float = 0.3,
    tol_spike_ms: float = 0.5,
    max_lag_ms: float = 40.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Remove duplicate units of a set greedily, keeping the best one of each group.

    The best unit is kept and every surviving unit scoring at least threshold with it
    is dropped; then the next best survivor, and so on. Kept units are pairwise below
    threshold, so any number of copies of one motor unit collapse to one.

    Args:
        spike_trains (SpikeTrains): Binary array (samples, n_units) or (samples,), or
            a list with one integer array of spike samples per unit.
        quality (np.ndarray): Per-unit quality, higher is better (e.g. SIL, or
            -CoV-ISI), shape (n_units,). NaN ranks last; ties keep unit order.
        fs (float): Sampling frequency in Hz.
        metric (SymmetricMetric): Score compared with threshold: "roa", "f1" or
            "match". Defaults to "roa".
        threshold (float): Score at or above which two units are duplicates.
            Defaults to 0.3.
        tol_spike_ms (float): Spike coincidence tolerance in ms. Defaults to 0.5.
        max_lag_ms (float): Largest lag searched between two units in ms. Defaults to 40.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]:
            - kept_idx: sorted input indices of the kept units.
            - duplicates_info: one row per dropped unit with kept_out_idx (kept unit
              in kept_idx), kept_in_idx and duplicate_in_idx (input indices), lag
              (samples added to the duplicate to align it), the metric score, and
              quality_kept and quality_duplicate.

    Raises:
        ValueError: If metric is not symmetric or quality has the wrong length.
    """
    if metric not in get_args(SymmetricMetric):
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {list(get_args(SymmetricMetric))}."
        )
    score, lag = compute_agreement_matrix(
        spike_trains, fs=fs, metric=metric, tol_spike_ms=tol_spike_ms, max_lag_ms=max_lag_ms
    )
    quality = np.asarray(quality, dtype=float)
    if quality.shape != (len(score),):
        raise ValueError(f"quality has shape {quality.shape}; expected ({len(score)},).")
    duplicate = score >= threshold
    np.fill_diagonal(duplicate, False)

    alive = np.ones(len(score), dtype=bool)
    kept_in, dup_in = [], []
    for unit in np.argsort(np.where(np.isnan(quality), np.inf, -quality), kind="stable"):
        if alive[unit]:
            dups = np.flatnonzero(duplicate[unit] & alive)
            alive[dups] = False
            kept_in += [unit] * len(dups)
            dup_in += dups.tolist()
    kept_idx = np.flatnonzero(alive)
    kept_in, dup_in = np.array(kept_in, dtype=int), np.array(dup_in, dtype=int)
    duplicates_info = pd.DataFrame(
        {
            "kept_out_idx": np.searchsorted(kept_idx, kept_in),
            "kept_in_idx": kept_in,
            "duplicate_in_idx": dup_in,
            "lag": lag[kept_in, dup_in],
            metric: score[kept_in, dup_in],
            "quality_kept": quality[kept_in],
            "quality_duplicate": quality[dup_in],
        }
    )
    return kept_idx, duplicates_info.sort_values(
        ["kept_out_idx", "duplicate_in_idx"], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Deprecated entry points: thin wrappers with the old arguments and outputs
# ---------------------------------------------------------------------------


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in a future release; use {new} instead.",
        FutureWarning,
        stacklevel=3,
    )


def _legacy_outputs(pairs: pd.DataFrame) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """(roa, pair_idx, pair_lag) of a pairs DataFrame, as returned by the old functions."""
    pair_idx = [(int(r), int(t)) for r, t in zip(pairs["ref_idx"], pairs["test_idx"], strict=True)]
    return pairs["roa"].to_numpy(), pair_idx, pairs["lag"].to_numpy(dtype=float)


def rate_of_agreement_paired(
    spike_trains_ref: SpikeTrains,
    spike_trains_test: SpikeTrains,
    fs: float = 2048,
    tol_spike_ms: float = 1,
    tol_train_ms: float = 40,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Deprecated: use `compare_paired_spikes`.

    Args:
        spike_trains_ref (SpikeTrains): Reference trains (see `compare_paired_spikes`).
        spike_trains_test (SpikeTrains): Test trains, same number of units.
        fs (float): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (float): Spike tolerance in ms. Defaults to 1.
        tol_train_ms (float): Largest lag between trains in ms. Defaults to 40.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]: RoA (n,), (unit, unit)
        pairs and lag (n,) of each unit.
    """
    _deprecated("rate_of_agreement_paired", "compare_paired_spikes")
    return _legacy_outputs(
        compare_paired_spikes(
            spike_trains_ref,
            spike_trains_test,
            fs=fs,
            tol_spike_ms=tol_spike_ms,
            max_lag_ms=tol_train_ms,
        )
    )


def rate_of_agreement(
    spike_trains_ref: Optional[SpikeTrains],
    spike_trains_test: SpikeTrains,
    fs: float = 2048,
    tol_spike_ms: float = 1,
    tol_train_ms: float = 40,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[int]]:
    """Deprecated: use `compare_spikes`, or `find_duplicates` within a set.

    Args:
        spike_trains_ref (SpikeTrains, optional): Reference trains. None finds the
            duplicates within the test set (`find_duplicates` with units ranked by
            index and its default threshold).
        spike_trains_test (SpikeTrains): Test trains.
        fs (float): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (float): Spike tolerance in ms. Defaults to 1.
        tol_train_ms (float): Largest lag between trains in ms. Defaults to 40.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], List[int]]: RoA, (ref, test) pairs
        (or (lower, higher) unit pairs within a set) and lag of each pair, ordered by
        the first unit of the pair.
    """
    kwargs = {"fs": fs, "tol_spike_ms": tol_spike_ms, "max_lag_ms": tol_train_ms}
    if spike_trains_ref is None:
        _deprecated("rate_of_agreement(None, ...)", "find_duplicates")
        n_units = len(_spike_times(spike_trains_test))
        _, info = find_duplicates(spike_trains_test, np.zeros(n_units), **kwargs)
        # (lower, higher) unit pairs; the lag aligns the higher unit to the lower one
        flip = info["kept_in_idx"] > info["duplicate_in_idx"]
        pairs = pd.DataFrame(
            {
                "ref_idx": info[["kept_in_idx", "duplicate_in_idx"]].min(axis=1),
                "test_idx": info[["kept_in_idx", "duplicate_in_idx"]].max(axis=1),
                "lag": info["lag"].where(~flip, -info["lag"]),
                "roa": info["roa"],
            }
        )
    else:
        _deprecated("rate_of_agreement", "compare_spikes")
        pairs = compare_spikes(spike_trains_ref, spike_trains_test, **kwargs)
    roa, pair_idx, pair_lag = _legacy_outputs(
        pairs.sort_values(["ref_idx", "test_idx"], ignore_index=True)
    )
    return roa, pair_idx, pair_lag.astype(int).tolist()


def rate_of_agreement_full(
    spike_trains_ref: SpikeTrains,
    spike_trains_test: SpikeTrains,
    fs: float = 2048,
    tol_spike_ms: float = 1,
    tol_train_ms: float = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated: use `compute_agreement_matrix`.

    Args:
        spike_trains_ref (SpikeTrains): Reference trains.
        spike_trains_test (SpikeTrains): Test trains.
        fs (float): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (float): Spike tolerance in ms. Defaults to 1.
        tol_train_ms (float): Largest lag between trains in ms. Defaults to 40.

    Returns:
        Tuple[np.ndarray, np.ndarray]: RoA and lag, each with shape (n_ref, n_test).
    """
    _deprecated("rate_of_agreement_full", "compute_agreement_matrix")
    roa, lag = compute_agreement_matrix(
        spike_trains_ref,
        spike_trains_test,
        fs=fs,
        tol_spike_ms=tol_spike_ms,
        max_lag_ms=tol_train_ms,
    )
    return roa, lag.astype(float)


def rate_of_agreement_all(
    spike_trains: SpikeTrains,
    fs: float = 2048,
    tol_spike_ms: float = 1,
    tol_train_ms: Optional[float] = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated: use `compute_agreement_matrix` with no test set.

    Args:
        spike_trains (SpikeTrains): Spike trains of one set.
        fs (float): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (float): Spike tolerance in ms. Defaults to 1.
        tol_train_ms (float, optional): Largest lag between trains in ms; None
            searches the whole recording (binary arrays only). Defaults to 40.

    Returns:
        Tuple[np.ndarray, np.ndarray]: RoA (diagonal 1) and lag, each with shape
        (n_units, n_units); shape (1,) for one unit and (0,) for no input.
    """
    _deprecated("rate_of_agreement_all", "compute_agreement_matrix")
    if len(spike_trains) < 1:
        return np.empty(0), np.empty(0)
    if tol_train_ms is None:
        tol_train_ms = 1000 * len(spike_trains) / fs
    roa, lag = compute_agreement_matrix(
        spike_trains, fs=fs, tol_spike_ms=tol_spike_ms, max_lag_ms=tol_train_ms
    )
    np.fill_diagonal(roa, 1)
    return (roa.ravel(), lag.ravel()) if len(roa) == 1 else (roa, lag)


def precision_sensitivity_f1_paired(
    spike_trains_ref: SpikeTrains,
    spike_trains_test: SpikeTrains,
    fs: float = 2048,
    tol_spike_ms: float = 1,
    tol_train_ms: float = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """Deprecated: use `compare_paired_spikes`.

    Args:
        spike_trains_ref (SpikeTrains): Reference trains.
        spike_trains_test (SpikeTrains): Test trains, same number of units.
        fs (float): Sampling frequency in Hz. Defaults to 2048.
        tol_spike_ms (float): Spike tolerance in ms. Defaults to 1.
        tol_train_ms (float): Largest lag between trains in ms. Defaults to 40.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], np.ndarray]:
        precision, sensitivity and F1 (n,), (unit, unit) pairs and lag (n,) of each unit.
    """
    _deprecated("precision_sensitivity_f1_paired", "compare_paired_spikes")
    pairs = compare_paired_spikes(
        spike_trains_ref,
        spike_trains_test,
        fs=fs,
        tol_spike_ms=tol_spike_ms,
        max_lag_ms=tol_train_ms,
    )
    _, pair_idx, pair_lag = _legacy_outputs(pairs)
    return (
        pairs["precision"].to_numpy(),
        pairs["sensitivity"].to_numpy(),
        pairs["f1"].to_numpy(),
        pair_idx,
        pair_lag,
    )
