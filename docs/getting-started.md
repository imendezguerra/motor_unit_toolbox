# Getting started

## Installation

--8<-- "README.md:install"

## Data conventions

- **Spike trains** are binary arrays of shape `(samples, motor units)`. A single unit can be
  passed as a 1D array of length `samples`.
- **Firings** are a list with one array of spike times (in samples) per motor unit. Convert
  between the two with `utils.firings_to_binary` and `utils.binary_to_firings`. The
  `spike_comp` functions accept either format.
- **MUAPs** are arrays of shape `(rows, cols, samples)` for one motor unit on an electrode grid.
  Several units are stacked as `(units, rows, cols, samples)`. For real recordings,
  `props.get_muaps(spike_trains, emg, fs)` computes them by spike-triggered averaging of the
  `(rows, cols, samples)` EMG.
- **Timestamps** are the time in seconds of each sample, e.g. `np.arange(n_samples) / fs`.

## Firing properties and spike train comparison

--8<-- "README.md:quickstart"

### Comparing decompositions

Compare two decompositions in two steps:

1. **Remove duplicates within each set.** `find_duplicates` keeps the best unit (highest
   `quality`, e.g. SIL) and drops every unit whose rate of agreement with it is at least
   `threshold`, so any number of copies of one motor unit collapse to one.
2. **Match the unique units across sets.** `compare_spikes` pairs them one-to-one (Hungarian
   algorithm on `match_metric`, RoA by default). Pairs without common spikes, or below
   `min_score`, are unmatched: a reference unit missing from the table was not found.
   `report_unmatched=True` adds a row for each unmatched test unit, with all its spikes as
   false positives.

```python exec="on" source="block" result="text" session="quickstart"
# Test decomposition: both units plus a copy of unit 0 delayed by 3 samples
decomposition = np.column_stack([spike_trains, np.roll(spike_trains[:, 0], 3)])

kept_idx, duplicates_info = spike_comp.find_duplicates(
    decomposition, quality=np.array([0.90, 0.95, 0.92]), fs=fs
)
print("Kept units:", kept_idx)
print(duplicates_info.to_string())

agreement = spike_comp.compare_spikes(spike_trains, decomposition[:, kept_idx], fs=fs)
print(agreement)
```

The reference set is the ground truth: `fn` counts reference spikes missed, `fp` test spikes
not in the reference, and `lag` is the shift (in samples) added to the test spikes to align
them. `compute_agreement_matrix` returns the score and lag of every pair.

## Tracking motor units across contractions

One approach to track MUs across contractions is to track their MUAPs. In this package, there are two approaches depending on the contraction type:

| Condition | MUAP shapes | Use | How it works |
|---|---|---|---|
| **Isometric** (fixed joint angle) | Stable across contractions | `muap_comp.cluster_muaps` | Pools every MUAP and groups them by agglomerative (hierarchical) clustering. The distance threshold is chosen automatically by the best silhouette score |
| **Dynamic** (changing joint angle or muscle length) | Drift gradually from one trial to the next | `muap_comp.assign_muaps_all_trials` | Matches units between *neighbouring* trials (Hungarian algorithm, under a distance threshold) and chains the matches. Units left unmatched are then compared across more distant trials |

The examples below use synthetic MUAPs on an 8×8 grid. Each motor unit has its own location on
the grid and its own waveform width:

```python exec="on" source="above" session="tracking"
import numpy as np

from motor_unit_toolbox import muap_comp

rng = np.random.default_rng(0)
rows, cols, n_samples = 8, 8, 64
r, c = np.mgrid[:rows, :cols]
t = np.arange(n_samples) - n_samples / 2


def make_muap(row, col, width):
    """Synthetic MUAP (rows, cols, samples): a spatial blob times a biphasic waveform."""
    spatial = np.exp(-((r - row) ** 2 + (c - col) ** 2) / 4)
    waveform = -t * np.exp(-(t / width) ** 2)
    return spatial[..., None] * waveform + rng.normal(0, 0.05, (rows, cols, n_samples))


# Four motor units: (row, column, waveform width)
units = [(1, 1, 4), (2, 6, 6), (6, 2, 5), (5, 5, 8)]
```

### Isometric contractions: clustering

Three contractions identify the units in a different order each time. Unit 3 is missing from
the second contraction. All MUAPs are pooled, their pairwise distances computed, and then
clustered:

```python exec="on" source="block" result="text" session="tracking"
contractions = [[0, 1, 2, 3], [2, 0, 1], [3, 1, 0, 2]]
true_units = np.concatenate(contractions)
muaps = np.stack([make_muap(*units[u]) for u in true_units])

# Compute muap distance (1-corr) across muaps and select the channels with outlier peak to peak value for comparison
dist, _ = muap_comp.compute_all_muaps_dist(
    muaps, dist_metric="corr", ch_sel_method="iqr", ch_sel_metric="ptp"
)

# Track MUAPs across conditions
opt, cluster_out = muap_comp.cluster_muaps(
    dist, cluster_method="ward", dist_metric="corr", ch_sel_method="iqr", ch_sel_metric="ptp",
    flag_plot=False,
)

# Select the labels of clustering threshold that maximises the clustering SIL
labels = cluster_out["labels"][opt["opt_idx"].iloc[0]].astype(int)
print("True units:", true_units)
print("Labels:    ", labels)  # same label = same motor unit
print(opt[["opt_thr", "opt_sil", "opt_n_clusters"]].round(3))
```

- `cluster_out` holds the labels and the clustering scores for every threshold tried
  (`thr_vals`).
- `flag_plot=True` (the default) plots the dendrogram and the scores against the threshold,
  which is useful to check the chosen threshold.
- Pooling MUAPs from the same contraction also flags **duplicates**: two units decomposed
  from one contraction that share a label are likely the same motor unit.

#### Keeping units from the same contraction apart

Units decomposed from the same contraction are normally different motor units. Pass
`trial_labels` to stop them from being clustered together. The distances between them are then
set above every threshold before clustering (`muap_comp.mask_within_trial_dist`):

```python exec="on" source="block" result="text" session="tracking"
trial_labels = np.repeat(np.arange(len(contractions)), [len(trial) for trial in contractions])

opt, cluster_out = muap_comp.cluster_muaps(
    dist, cluster_method="complete", trial_labels=trial_labels, flag_plot=False
)
labels = cluster_out["labels"][opt["opt_idx"].iloc[0]].astype(int)
print("True units:", true_units)
print("Labels:    ", labels)  # 4 units, at most one per contraction in each
```

- Only `cluster_method="complete"` guarantees the separation. Its cluster distance is the
  *largest* pairwise distance, so a cluster can never contain two units from the same contraction.
- Other linkage methods average distances, so they can still merge two such units through a
  unit from another contraction. After clustering, `cluster_muaps` checks the selected threshold
  and raises a warning naming the threshold, the cluster, the units and their contraction.
- The clustering scores (silhouette, etc.) are always computed on the original distances.
- With `trial_labels`, duplicates are no longer flagged: a duplicate ends up in a cluster of its own.

### Dynamic contractions: sequential assignment

Five trials are recorded at increasing joint angles. Between trials, each unit's MUAP moves
across the grid and widens a little. Neighbouring trials look alike, but the first and last
trials do not:

```python exec="on" source="block" result="text" session="tracking"
n_trials = 5
order = [rng.permutation(len(units)) for _ in range(n_trials)]   # shuffled unit order per trial
muaps = np.stack([
    make_muap(units[u][0] + 0.6 * k, units[u][1] - 0.4 * k, units[u][2] * (1 + 0.15 * k))
    for k in range(n_trials) for u in order[k]
])
trial_labels = np.repeat(np.arange(n_trials), len(units))         # trial of each MUAP

group_labels, group_sets, links, graph = muap_comp.assign_muaps_all_trials(
    muaps,
    trial_labels,
    trial_set=list(range(n_trials)),   # trial order: neighbours are matched first
    dist_metric="nmse",
    dist_thr=0.3,                      # pairs further apart than this are never linked
)
print("True units:  ", np.concatenate(order))
print("Group labels:", group_labels.astype(int))
```

- `group_labels` gives one motor unit label per MUAP.
- `group_sets` lists the MUAP indices of each tracked unit.
- `links` is a table of every accepted match and its distance.
- `graph` is the underlying `networkx` graph of the links.

For comparison, clustering the same drifting MUAPs with `cluster_muaps` finds 5 clusters for 4
units. It splits two units across clusters and puts MUAPs of both into the same cluster. Once a
unit's MUAP has drifted far from its first trial, it no longer looks like itself. Linking
neighbouring trials is what keeps the track.
