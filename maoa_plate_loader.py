from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd


# ------------------ SPIKE LIST LOADER ------------------ #

def load_spike_list(path: str) -> pd.DataFrame:
    """
    Load and clean an Axion *_spike_list.csv file.

    Returns columns:
      - time_s
      - electrode  (e.g., 'A1_11')
      - well       (e.g., 'A1')
      - electrode_idx (Int64)
      - amplitude_mV
    """
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["Time (s)"]).copy()

    df = df.rename(
        columns={
            "Time (s)": "time_s",
            "Electrode": "electrode",
            "Amplitude (mV)": "amplitude_mV",
        }
    )

    well_elec = df["electrode"].astype(str).str.extract(
        r"^(?P<well>[A-Z]\d+)_(?P<electrode_idx>\d+)$"
    )
    df["well"] = well_elec["well"]
    df["electrode_idx"] = pd.to_numeric(
        well_elec["electrode_idx"], errors="coerce"
    ).astype("Int64")

    return df[["time_s", "electrode", "well", "electrode_idx", "amplitude_mV"]]


# ------------------ ELECTRODE BURSTS (100 ms, 5 spikes) ------------------ #

def detect_bursts_for_electrode(
    times: np.ndarray,
    isi_max: float = 0.1,      # 100 ms
    min_spikes: int = 5,       # minimum spikes per burst
):
    """
    ISI-based electrode burst detector.

    A burst is a sequence of >= min_spikes spikes such that all
    consecutive ISIs <= isi_max.
    Returns list of arrays of spike times (one per burst).
    """
    times = np.sort(np.asarray(times, float))
    n = len(times)
    if n < min_spikes:
        return []

    bursts = []
    current_idx = [0]

    for i in range(1, n):
        isi = times[i] - times[i - 1]
        if isi <= isi_max:
            current_idx.append(i)
        else:
            if len(current_idx) >= min_spikes:
                bursts.append(times[current_idx[0]: current_idx[-1] + 1])
            current_idx = [i]

    if len(current_idx) >= min_spikes:
        bursts.append(times[current_idx[0]: current_idx[-1] + 1])

    return bursts


def compute_electrode_metrics(
    spikes_df: pd.DataFrame,
    isi_max: float = 0.1,
    min_spikes_per_burst: int = 5,
) -> tuple[pd.DataFrame, float]:
    """
    Per-electrode spike & burst rates.

    Returns:
      elec_df with columns:
        ['electrode', 'well', 'n_spikes', 'spike_rate_hz',
         'n_bursts', 'burst_rate_hz']
      duration_s (recording length)
    """
    t_min = spikes_df["time_s"].min()
    t_max = spikes_df["time_s"].max()
    duration = float(t_max - t_min) if t_max > t_min else float("nan")

    rows = []
    for elec, g in spikes_df.groupby("electrode"):
        times = g["time_s"].values
        n_spikes = len(times)
        spike_rate = n_spikes / duration if duration > 0 else float("nan")

        bursts = detect_bursts_for_electrode(
            times,
            isi_max=isi_max,
            min_spikes=min_spikes_per_burst,
        )
        n_bursts = len(bursts)
        burst_rate = n_bursts / duration if duration > 0 else float("nan")

        rows.append(
            {
                "electrode": elec,
                "well": g["well"].iloc[0],
                "n_spikes": n_spikes,
                "spike_rate_hz": spike_rate,
                "n_bursts": n_bursts,
                "burst_rate_hz": burst_rate,
            }
        )

    elec_df = pd.DataFrame(rows)
    return elec_df, duration


# ------------------ NETWORK BURSTS (100 ms, 50 spikes, 35% elecs) ------------------ #

def detect_network_bursts_for_well(
    spikes_df_well: pd.DataFrame,
    isi_max_network: float = 0.1,         # 100 ms
    min_spikes_network: int = 50,         # total spikes in burst
    min_participating_pct: float = 0.35,  # 35% electrodes
):
    """
    Network bursts for ONE well, ISI-based.

    Pool spikes across electrodes; candidate bursts are runs of spikes
    where ISI <= isi_max_network. Keep candidates with:
      - >= min_spikes_network spikes
      - at least min_participating_pct of electrodes contributing.
    """
    if len(spikes_df_well) == 0:
        return []

    times = spikes_df_well["time_s"].to_numpy(float)
    elecs = spikes_df_well["electrode"].astype(str).to_numpy()
    order = np.argsort(times)
    times = times[order]
    elecs = elecs[order]

    n = len(times)
    if n < min_spikes_network:
        return []

    total_electrodes = len(np.unique(elecs))
    if total_electrodes == 0:
        return []

    bursts = []
    current_idx = [0]

    def maybe_add(idxs):
        if len(idxs) < min_spikes_network:
            return
        participating = np.unique(elecs[idxs])
        frac = len(participating) / total_electrodes
        if frac >= min_participating_pct:
            bursts.append(
                {
                    "start_time_s": float(times[idxs[0]]),
                    "end_time_s": float(times[idxs[-1]]),
                    "n_spikes": int(len(idxs)),
                    "n_participating_electrodes": int(len(participating)),
                    "participating_fraction": float(frac),
                }
            )

    for i in range(1, n):
        isi = times[i] - times[i - 1]
        if isi <= isi_max_network:
            current_idx.append(i)
        else:
            maybe_add(current_idx)
            current_idx = [i]
    maybe_add(current_idx)

    return bursts


def compute_well_network_metrics(
    spikes_df: pd.DataFrame,
    elec_df: pd.DataFrame,
    duration_s: float,
    isi_max_network: float = 0.1,
    min_spikes_network: int = 50,
    min_participating_pct: float = 0.35,
):
    """
    Per-well spike/burst rates + network event rates.

    Returns DataFrame with columns:
      ['well', 'well_spike_rate_hz', 'well_burst_rate_hz',
       'n_network_events', 'network_event_rate_hz']
    """
    rows = []
    for well, g in spikes_df.groupby("well"):
        n_spikes = len(g)
        well_spike_rate = n_spikes / duration_s if duration_s > 0 else float("nan")

        n_bursts = elec_df.loc[elec_df["well"] == well, "n_bursts"].sum()
        well_burst_rate = n_bursts / duration_s if duration_s > 0 else float("nan")

        net_bursts = detect_network_bursts_for_well(
            g,
            isi_max_network=isi_max_network,
            min_spikes_network=min_spikes_network,
            min_participating_pct=min_participating_pct,
        )
        n_events = len(net_bursts)
        net_rate = n_events / duration_s if duration_s > 0 else float("nan")

        rows.append(
            {
                "well": well,
                "well_spike_rate_hz": well_spike_rate,
                "well_burst_rate_hz": well_burst_rate,
                "n_network_events": n_events,
                "network_event_rate_hz": net_rate,
            }
        )

    return pd.DataFrame(rows)


# ------------------ SYNCHRONY INDEX (AREA UNDER NORM. XCORR) ------------------ #

def _make_binned_train(times, t_min, t_max, bin_size):
    if t_max <= t_min:
        return np.array([], dtype=float)
    n_bins = int(np.ceil((t_max - t_min) / bin_size))
    if n_bins <= 0:
        return np.array([], dtype=float)
    edges = t_min + np.arange(n_bins + 1) * bin_size
    counts, _ = np.histogram(times, bins=edges)
    return counts.astype(float)


def _normalized_xcorr(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.array([0.0])
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_z = x - x.mean()
    y_z = y - y.mean()
    denom = np.sqrt((x_z**2).sum() * (y_z**2).sum())
    if denom == 0:
        return np.zeros(2 * len(x) - 1)
    return np.correlate(x_z, y_z, mode="full") / denom


def compute_synchrony_index_per_well(
    spikes_df: pd.DataFrame,
    synchrony_window_s: float = 0.02,   # 20 ms
    bin_size: float | None = None,
    max_lag_s: float | None = None,
    min_spikes_per_electrode: int = 10,
):
    """
    For each well: one synchrony value:

        synchrony_index = mean area under normalized cross-correlation
        across all electrode pairs, within ±max_lag_s.

    Returns DataFrame with:
      ['well', 'synchrony_index', 'n_electrodes_for_sync']
    """
    if bin_size is None:
        bin_size = synchrony_window_s
    if max_lag_s is None:
        max_lag_s = synchrony_window_s

    rows = []

    for well, g in spikes_df.groupby("well"):
        elec_groups = {
            elec: grp["time_s"].to_numpy(float)
            for elec, grp in g.groupby("electrode")
            if len(grp) >= min_spikes_per_electrode
        }
        electrodes = list(elec_groups.keys())
        n_elecs = len(electrodes)

        if n_elecs < 2:
            rows.append(
                {
                    "well": well,
                    "synchrony_index": np.nan,
                    "n_electrodes_for_sync": n_elecs,
                }
            )
            continue

        t_min = g["time_s"].min()
        t_max = g["time_s"].max()

        trains = {
            elec: _make_binned_train(times, t_min, t_max, bin_size)
            for elec, times in elec_groups.items()
        }

        L = len(next(iter(trains.values())))
        if L == 0:
            rows.append(
                {
                    "well": well,
                    "synchrony_index": np.nan,
                    "n_electrodes_for_sync": n_elecs,
                }
            )
            continue

        lags = np.arange(-L + 1, L) * bin_size
        lag_mask = np.abs(lags) <= max_lag_s

        areas = []
        for i in range(n_elecs):
            for j in range(i + 1, n_elecs):
                x = trains[electrodes[i]]
                y = trains[electrodes[j]]
                norm_corr = _normalized_xcorr(x, y)
                norm_win = norm_corr[lag_mask]
                area = norm_win.sum() * bin_size
                areas.append(area)

        sync_index = float(np.nanmean(areas)) if len(areas) else np.nan

        rows.append(
            {
                "well": well,
                "synchrony_index": sync_index,
                "n_electrodes_for_sync": n_elecs,
            }
        )

    return pd.DataFrame(rows)


# ------------------ TOP-LEVEL FUNCTION (WHAT YOU ASKED FOR) ------------------ #

def summarize_spike_file_to_well_df(
    spike_list_path: str,
    return_electrode_metrics: bool = False,
):
    """
    INPUT:
      spike_list_path: path to one *_spike_list.csv (one plate × timepoint)

    OUTPUT:
      pandas DataFrame with ONE ROW PER WELL, columns:

        - well
        - firing_rate_stats_hz      -> [mean, std] across electrodes
        - burst_rate_stats_hz       -> [mean, std] across electrodes
        - network_event_rate_hz
        - synchrony_index
        - electrodes                -> list of electrode IDs in that well
        - firing_rate_vector_hz     -> list of per-electrode firing rates
        - (optional) electrode_metrics_dict
              dict: electrode -> {spike_rate_hz, burst_rate_hz, n_spikes, n_bursts}
    """
    # Load & compute all the building blocks
    spikes = load_spike_list(spike_list_path)
    elec_df, duration_s = compute_electrode_metrics(spikes)
    well_net_df = compute_well_network_metrics(spikes, elec_df, duration_s)
    sync_df = compute_synchrony_index_per_well(spikes)

    rows = []
    for well in sorted(spikes["well"].unique()):
        elec_sub = elec_df[elec_df["well"] == well]

        spike_rates = elec_sub["spike_rate_hz"].to_numpy(float)
        burst_rates = elec_sub["burst_rate_hz"].to_numpy(float)
        electrodes = elec_sub["electrode"].tolist()

        well_row = well_net_df[well_net_df["well"] == well].iloc[0]
        sync_row = sync_df[sync_df["well"] == well].iloc[0]

        if return_electrode_metrics:
            elec_dict = {
                r["electrode"]: {
                    "spike_rate_hz": float(r["spike_rate_hz"]),
                    "burst_rate_hz": float(r["burst_rate_hz"]),
                    "n_spikes": int(r["n_spikes"]),
                    "n_bursts": int(r["n_bursts"]),
                }
                for _, r in elec_sub.iterrows()
            }
        else:
            elec_dict = None

        rows.append(
            {
                "well": well,
                "firing_rate_stats_hz": [
                    float(np.nanmean(spike_rates)) if len(spike_rates) else np.nan,
                    float(np.nanstd(spike_rates)) if len(spike_rates) else np.nan,
                ],
                "burst_rate_stats_hz": [
                    float(np.nanmean(burst_rates)) if len(burst_rates) else np.nan,
                    float(np.nanstd(burst_rates)) if len(burst_rates) else np.nan,
                ],
                "network_event_rate_hz": float(well_row["network_event_rate_hz"]),
                "synchrony_index": float(sync_row["synchrony_index"]),
                "electrodes": electrodes,
                "firing_rate_vector_hz": spike_rates.tolist(),
                # if you also want per-electrode burst rates as a vector:
                # "burst_rate_vector_hz": burst_rates.tolist(),
                "electrode_metrics_dict": elec_dict,
            }
        )

    return pd.DataFrame(rows)
