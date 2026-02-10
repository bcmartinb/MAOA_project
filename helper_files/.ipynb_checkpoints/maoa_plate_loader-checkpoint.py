from __future__ import annotations

import os
import glob
from datetime import datetime, date

import numpy as np
import pandas as pd

# nice progress bars
try:
    from tqdm.auto import tqdm
except Exception:  # if tqdm isn't installed
    def tqdm(x, **kwargs):
        return x


# =====================================================================
# PLATE METADATA
# =====================================================================

# NOTE:
# - origin_date = biological "birth" date of the organoids
# - organoid_age_days = (recording_date - origin_date).days
#
# BATCH 1: MAOA1 / MAOA2 / MAOA3
#   "were 53 days old on March 25, 2025"
#   → origin_date = 2025-03-25 - 53 days = 2025-01-31
#
# BATCH 2: MAOA72 / MAOA76 / MAOA77 (June recordings)
#   "4 months old on June 13, 2025" (~120 days)
#   → origin_date = 2025-06-13 - 120 days = 2025-02-13
#
# BATCH 3: MAOA72 / MAOA76 / MAOA77 (Sept–Oct recordings, same batch)
#   "6 months old on August 12, 2025" (~180 days)
#   → also gives origin_date = 2025-02-13 (consistent with batch 2)

PLATE_METADATA: dict[str, dict] = {
    # -------------------- FIRST THREE PLATES (batch 1) --------------------
    "MAOA1": {
        "origin_date": date(2025, 1, 31),
        "wells_to_cell_line": {
            # row A
            "A1": "C3",  "A2": "C30", "A3": "GM", "A4": "C90",
            # row B
            "B1": "C71", "B2": "C30", "B3": "GM", "B4": "H9",
            # row C
            "C1": "C71", "C2": "C12", "C3": "GM", "C4": "C54",
        },
    },

    "MAOA2": {
        "origin_date": date(2025, 1, 31),
        "wells_to_cell_line": {
            "A1": "C3",  "A2": "C30", "A3": "GM", "A4": "C90",
            "B1": "C71", "B2": "C30", "B3": "GM", "B4": "H9",
            "C1": "C71", "C2": "C12", "C3": "GM", "C4": "C54",
        },
    },

    "MAOA3": {
        "origin_date": date(2025, 1, 31),
        "wells_to_cell_line": {
            "A1": "C3",  "A2": "C30", "A3": "GM", "A4": "C54",
            "B1": "C90", "B2": "C12", "B3": "GM", "B4": "C71",
            "C1": "C90", "C2": "C12", "C3": "GM", "C4": "H9",
        },
    },

    # -------------------- JUNE BATCH (plates 72 / 76 / 77) ---------------
    # Layout is the same for all three in this batch
    "MAOA72": {
        "origin_date": date(2025, 2, 13),
        "wells_to_cell_line": {
            "A1": "C90m", "A2": "C90m", "A3": "C90m", "A4": "C90m",
            "B1": "C12c", "B2": "C12c", "B3": "C12c", "B4": "C12c",
            "C1": "GM",  "C2": "GM",  "C3": "GM",  "C4": "GM",
        },
    },

    "MAOA76": {
        "origin_date": date(2025, 2, 13),
        "wells_to_cell_line": {
            "A1": "C90m", "A2": "C90m", "A3": "C90m", "A4": "C90m",
            "B1": "C12c", "B2": "C12c", "B3": "C12c", "B4": "C12c",
            "C1": "GM",  "C2": "GM",  "C3": "GM",  "C4": "GM",
        },
    },

    "MAOA77": {
        "origin_date": date(2025, 2, 13),
        "wells_to_cell_line": {
            "A1": "C90m", "A2": "C90m", "A3": "C90m", "A4": "C90m",
            "B1": "C12c", "B2": "C12c", "B3": "C12c", "B4": "C12c",
            "C1": "GM",  "C2": "GM",  "C3": "GM",  "C4": "GM",
        },
    },

    # -------------------- SEPT–OCT BATCH (same biological batch) ---------
    # Same plate IDs as above but with different layouts starting Sept 25
    "MAOA72_SEP": {  # internal logical name – we’ll handle mapping below
        "origin_date": date(2025, 2, 13),
        "wells_to_cell_line": {
            "A1": "C3",  "A2": "C71", "A3": "C90", "A4": "GM",
            "B1": "C3",  "B2": "C71", "B3": "C90", "B4": "GM",
            "C1": "C12", "C2": "C30", "C3": "C54", "C4": "C30",
        },
    },

    "MAOA76_SEP": {
        "origin_date": date(2025, 2, 13),
        "wells_to_cell_line": {
            "A1": "C3",  "A2": "C71", "A3": "C90", "A4": "GM",   # GM DETACHED
            "B1": "C3",  "B2": "C71", "B3": "C90", "B4": "GM",
            "C1": "C12", "C2": "C30", "C3": "C54", "C4": "C12",
        },
    },

    "MAOA77_SEP": {
        "origin_date": date(2025, 2, 13),
        "wells_to_cell_line": {
            "A1": "C3",  "A2": "C71", "A3": "C90", "A4": "GM",
            "B1": "C3",  "B2": "C71", "B3": "C90", "B4": "GM",
            "C1": "C12", "C2": "C30", "C3": "C54", "C4": "C12",   # C12 DETACHED in C1
        },
    },
}

# For now we’ll keep it simple:
# - Use MAOA72 / 76 / 77 layouts for all dates up to 2025-09-24
# - Use *_SEP layouts for dates >= 2025-09-25
# If you later want this to branch by day, we can refine.

SEP_LAYOUT_SWITCH_DATE = date(2025, 9, 25)


def _get_plate_key_for_date(plate_name: str, rec_date: date) -> str:
    """Internal: choose which metadata block to use for plates that changed layout."""
    if plate_name in {"MAOA72", "MAOA76", "MAOA77"} and rec_date >= SEP_LAYOUT_SWITCH_DATE:
        return f"{plate_name}_SEP"
    return plate_name


# =====================================================================
# ELECTRODE & SPIKE HELPERS
# =====================================================================

def get_expected_electrodes_for_well(well: str) -> list[str]:
    """
    Return the full set of expected electrode IDs for a well,
    assuming an 8x8 Axion-style grid:

        rows: 1..8, cols: 1..8
        indices: 11-18, 21-28, ..., 81-88

    e.g. well 'A1' → ['A1_11', 'A1_12', ..., 'A1_88']
    """
    electrodes = []
    for row in range(1, 9):
        for col in range(1, 9):
            idx = f"{row}{col}"
            electrodes.append(f"{well}_{idx}")
    return electrodes


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


# =====================================================================
# ELECTRODE BURSTS (ISI-BASED) & METRICS
# =====================================================================

def detect_bursts_for_electrode(
    times: np.ndarray,
    isi_max: float = 0.1,   # 100 ms
    min_spikes: int = 5,
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
      duration_s (recording length, from min to max time)
    """
    t_min = spikes_df["time_s"].min()
    t_max = spikes_df["time_s"].max()
    duration = float(t_max - t_min) if t_max > t_min else float("nan")

    rows = []
    for elec, g in spikes_df.groupby("electrode"):
        times = g["time_s"].values
        n_spikes = len(times)

        spike_rate = n_spikes / duration if duration > 0 else np.nan

        bursts = detect_bursts_for_electrode(
            times,
            isi_max=isi_max,
            min_spikes=min_spikes_per_burst,
        )
        n_bursts = len(bursts)
        burst_rate = n_bursts / duration if duration > 0 else np.nan

        rows.append(
            {
                "electrode": elec,
                "well": g["well"].iloc[0],
                "n_spikes": n_spikes,
                "spike_rate_hz": float(spike_rate),
                "n_bursts": n_bursts,
                "burst_rate_hz": float(burst_rate),
            }
        )

    elec_df = pd.DataFrame(rows)
    return elec_df, duration


# =====================================================================
# NETWORK BURSTS (ISI-BASED, PER WELL)
# =====================================================================

def detect_network_bursts_for_well(
    spikes_df_well: pd.DataFrame,
    isi_max_network: float = 0.1,         # 100 ms
    min_spikes_network: int = 50,         # total spikes
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

    def maybe_add(idxs: list[int]):
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
) -> pd.DataFrame:
    """
    Per-well spike/burst rates + network event rates.

    Returns DataFrame with columns:
      ['well', 'well_spike_rate_hz', 'well_burst_rate_hz',
       'n_network_events', 'network_event_rate_hz']
    """
    rows = []
    for well, g in spikes_df.groupby("well"):
        n_spikes = len(g)
        well_spike_rate = n_spikes / duration_s if duration_s > 0 else np.nan

        n_bursts = elec_df.loc[elec_df["well"] == well, "n_bursts"].sum()
        well_burst_rate = n_bursts / duration_s if duration_s > 0 else np.nan

        net_bursts = detect_network_bursts_for_well(
            g,
            isi_max_network=isi_max_network,
            min_spikes_network=min_spikes_network,
            min_participating_pct=min_participating_pct,
        )
        n_events = len(net_bursts)
        net_rate = n_events / duration_s if duration_s > 0 else np.nan

        rows.append(
            {
                "well": well,
                "well_spike_rate_hz": float(well_spike_rate),
                "well_burst_rate_hz": float(well_burst_rate),
                "n_network_events": int(n_events),
                "network_event_rate_hz": float(net_rate),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# SYNCHRONY (FAST, ZERO-LAG PEARSON)
# =====================================================================

def compute_synchrony_index_per_well(
    spikes_df: pd.DataFrame,
    synchrony_window_s: float = 0.02,   # 20 ms
    bin_size: float | None = None,
    min_spikes_per_electrode: int = 10,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Synchrony = mean zero-lag normalized cross-correlation (Pearson r)
    between all pairs of binned spike trains in a well.

    Steps per well:
      1) Bin spikes for each electrode with bin_size (default = synchrony_window_s)
      2) Z-score each binned train
      3) For each electrode pair, compute r(0)
      4) Synchrony index = mean r(0) across all pairs
    """
    if bin_size is None:
        bin_size = synchrony_window_s

    rows = []

    well_groups = spikes_df.groupby("well")
    iterator = well_groups
    if show_progress:
        iterator = tqdm(well_groups, desc="Computing synchrony (per well)")

    for well, g in iterator:
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
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            rows.append(
                {
                    "well": well,
                    "synchrony_index": np.nan,
                    "n_electrodes_for_sync": n_elecs,
                }
            )
            continue

        n_bins = int(np.ceil((t_max - t_min) / bin_size))
        if n_bins <= 0:
            rows.append(
                {
                    "well": well,
                    "synchrony_index": np.nan,
                    "n_electrodes_for_sync": n_elecs,
                }
            )
            continue

        edges = t_min + np.arange(n_bins + 1) * bin_size

        trains = {}
        for elec, times in elec_groups.items():
            counts, _ = np.histogram(times, bins=edges)
            trains[elec] = counts.astype(float)

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

        rs = []
        for i in range(n_elecs):
            x = trains[electrodes[i]]
            x_z = x - x.mean()
            x_den = np.sqrt((x_z ** 2).sum())
            if x_den == 0:
                continue
            for j in range(i + 1, n_elecs):
                y = trains[electrodes[j]]
                y_z = y - y.mean()
                y_den = np.sqrt((y_z ** 2).sum())
                if y_den == 0:
                    continue
                r0 = float((x_z * y_z).sum() / (x_den * y_den))
                rs.append(r0)

        sync_index = float(np.nanmean(rs)) if rs else np.nan

        rows.append(
            {
                "well": well,
                "synchrony_index": sync_index,
                "n_electrodes_for_sync": n_elecs,
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# SINGLE SPIKE FILE → WELL SUMMARY
# =====================================================================

def summarize_spike_file_to_well_df(
    spike_list_path: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Summarize one *_spike_list.csv (one plate × timepoint).

    OUTPUT DataFrame: one row per well, with columns:

      - well
      - firing_rate_stats_hz   [mean, std] over electrodes with n_spikes > 0
      - burst_rate_stats_hz    [mean, std] over electrodes with n_bursts > 0
      - network_event_rate_hz
      - synchrony_index
      - electrodes             list of all expected electrodes in the well
      - firing_rate_vector_hz  per-electrode rates (NaN if no spikes)
      - burst_rate_vector_hz   per-electrode rates (NaN if no bursts)
    """
    spikes = load_spike_list(spike_list_path)
    elec_df, duration_s = compute_electrode_metrics(spikes)
    well_net_df = compute_well_network_metrics(spikes, elec_df, duration_s)
    sync_df = compute_synchrony_index_per_well(
        spikes,
        synchrony_window_s=0.02,
        show_progress=show_progress,
    )

    rows = []

    wells = sorted(spikes["well"].unique())
    well_iter = wells
    if show_progress:
        well_iter = tqdm(wells, desc="Summarizing wells")

    for well in well_iter:
        expected_electrodes = get_expected_electrodes_for_well(well)
        expected_df = pd.DataFrame({"electrode": expected_electrodes})

        elec_sub = expected_df.merge(
            elec_df[
                ["electrode", "well", "n_spikes", "n_bursts",
                 "spike_rate_hz", "burst_rate_hz"]
            ],
            on="electrode",
            how="left",
        )

        n_spikes = elec_sub["n_spikes"]
        n_bursts = elec_sub["n_bursts"]

        spike_rates_all = elec_sub["spike_rate_hz"].astype(float).to_numpy()
        burst_rates_all = elec_sub["burst_rate_hz"].astype(float).to_numpy()

        # Means/SDs:
        # - firing stats over electrodes that have at least 1 spike
        # - burst stats over electrodes that have at least 1 burst
        spiking_mask = n_spikes.fillna(0).to_numpy() > 0
        bursting_mask = n_bursts.fillna(0).to_numpy() > 0

        if spiking_mask.any():
            firing_mean = float(np.nanmean(spike_rates_all[spiking_mask]))
            firing_std = float(np.nanstd(spike_rates_all[spiking_mask]))
        else:
            firing_mean = np.nan
            firing_std = np.nan

        if bursting_mask.any():
            burst_mean = float(np.nanmean(burst_rates_all[bursting_mask]))
            burst_std = float(np.nanstd(burst_rates_all[bursting_mask]))
        else:
            burst_mean = np.nan
            burst_std = np.nan

        electrodes = elec_sub["electrode"].tolist()

        firing_vec = spike_rates_all.copy()
        burst_vec = burst_rates_all.copy()

        # no spikes → NaN in firing vector
        no_spike_mask = n_spikes.isna() | (n_spikes == 0)
        firing_vec[no_spike_mask.to_numpy()] = np.nan

        # no bursts → NaN in burst vector
        no_burst_mask = n_bursts.isna() | (n_bursts == 0)
        burst_vec[no_burst_mask.to_numpy()] = np.nan

        well_row = well_net_df[well_net_df["well"] == well].iloc[0]
        sync_row = sync_df[sync_df["well"] == well].iloc[0]

        rows.append(
            {
                "well": well,
                "firing_rate_stats_hz": [firing_mean, firing_std],
                "burst_rate_stats_hz": [burst_mean, burst_std],
                "network_event_rate_hz": float(well_row["network_event_rate_hz"]),
                "synchrony_index": float(sync_row["synchrony_index"]),
                "electrodes": electrodes,
                "firing_rate_vector_hz": firing_vec.tolist(),
                "burst_rate_vector_hz": burst_vec.tolist(),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# CELL LINE + AGE HELPER
# =====================================================================

def get_cell_line_and_age(
    plate_name: str,
    well: str,
    recording_date: date,
):
    """
    Look up cell line and organoid age (days) for a given plate & well.

    Uses PLATE_METADATA and origin_date.
    Handles plates whose layout changed over time (72/76/77).
    """
    plate_key = _get_plate_key_for_date(plate_name, recording_date)
    meta = PLATE_METADATA.get(plate_key)
    if meta is None:
        return None, np.nan

    origin_date = meta.get("origin_date")
    cell_map = meta.get("wells_to_cell_line", {})

    cell_line = cell_map.get(well)

    if isinstance(origin_date, date):
        age_days = (recording_date - origin_date).days
    else:
        age_days = np.nan

    return cell_line, age_days


# =====================================================================
# WALK ALL DIRECTORIES → BIG DATAFRAME
# =====================================================================

def summarize_all_spike_files(
    root_dir: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Walk a directory tree like:

        root_dir/
          04.14.2025/
            MAOA1/
              ..._spike_list.csv
            MAOA2/
              ..._spike_list.csv
            ...
          04.18.2025/
            MAOA1/ ...
            ...

    For every *_spike_list.csv file:
      - run summarize_spike_file_to_well_df()
      - add:
            recording_date
            recording_date_str
            plate_name
            cell_line
            organoid_age_days
            spike_list_path

    Returns a big DataFrame with one row per well × recording.

    Progress bars:
      - outer tqdm over date directories if show_progress=True
    """
    all_dfs: list[pd.DataFrame] = []

    date_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    date_dirs = sorted(date_dirs)

    date_iter = date_dirs
    if show_progress:
        date_iter = tqdm(date_dirs, desc="Recording date folders")

    for date_str in date_iter:
        date_path = os.path.join(root_dir, date_str)

        # parse folder name as date like "04.14.2025"
        try:
            rec_date = datetime.strptime(date_str, "%m.%d.%Y").date()
        except Exception:
            # skip non-date folders
            continue

        plate_dirs = [
            d for d in os.listdir(date_path)
            if os.path.isdir(os.path.join(date_path, d))
        ]

        for plate_name in plate_dirs:
            plate_path = os.path.join(date_path, plate_name)

            spike_files = glob.glob(os.path.join(plate_path, "*_spike_list.csv"))
            if not spike_files:
                continue

            if show_progress:
                tqdm.write(f"Processing {date_str} / {plate_name} ({len(spike_files)} files)")

            for spike_file in spike_files:
                df_wells = summarize_spike_file_to_well_df(
                    spike_file,
                    show_progress=False,
                )

                cell_lines = []
                ages = []
                for well in df_wells["well"]:
                    cell_line, age_days = get_cell_line_and_age(
                        plate_name=plate_name,
                        well=well,
                        recording_date=rec_date,
                    )
                    cell_lines.append(cell_line)
                    ages.append(age_days)

                df_wells["recording_date"] = rec_date
                df_wells["recording_date_str"] = date_str
                df_wells["plate_name"] = plate_name
                df_wells["cell_line"] = cell_lines
                df_wells["organoid_age_days"] = ages
                df_wells["spike_list_path"] = spike_file

                all_dfs.append(df_wells)

    if not all_dfs:
        return pd.DataFrame()

    big_df = pd.concat(all_dfs, ignore_index=True)
    return big_df


def assign_experiment(plate, rec_date):
    # Convert rec_date to date if string
    if isinstance(rec_date, str):
        rec_date = datetime.strptime(rec_date, "%m.%d.%Y").date()

    # Experiment 1: MAOA1–3
    if plate in ["MAOA1", "MAOA2", "MAOA3"]:
        return "Exp1"

    # Experiment 2: MAOA72/76/77 before Sept 25
    if plate in ["MAOA72", "MAOA76", "MAOA77"] and rec_date < date(2025,9,25):
        return "Exp2"

    # Experiment 3: MAOA72/76/77 on or after Sept 25
    if plate in ["MAOA72", "MAOA76", "MAOA77"] and rec_date >= date(2025,9,25):
        return "Exp3"

    return "Unknown"

#### 
