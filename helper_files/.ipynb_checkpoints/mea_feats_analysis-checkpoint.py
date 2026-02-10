import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.stats import ttest_ind



# Extract vectors
def expand_electrodes(row):
    return pd.DataFrame({
        "plate_name": row["plate_name"],
        "well": row["well"],
        "cell_line": row["cell_line"],
        "recording_date": row["recording_date"],
        "age_days": row["organoid_age_days"],
        "electrode": row["electrodes"],
        "firing_rate_hz": row["firing_rate_vector_hz"],
    })


def plot_plate_map(df, plate_name, use_group=False, double_label=False):
    """
    Draw a 3×4 plate map.

    Parameters
    ----------
    df : DataFrame
    plate_name : str
    use_group : bool
        If True → wells are colored by group (Mutant / WT / GM)
    double_label : bool
        If True → show both group AND cell_line inside each well

    Behavior
    --------
    - Default:     color by cell_line, label cell_line only
    - use_group:   color by group, label group only
    - double_label: color by group, label group + (cell_line)
    """

    sub = df[df["plate_name"] == plate_name].copy()
    if sub.empty:
        print(f"No data found for plate {plate_name}")
        return

    # one row per well
    sub = sub.drop_duplicates(subset=["well"]).copy()

    # parse coordinates
    sub["row"] = sub["well"].str[0]
    sub["col"] = sub["well"].str[1:].astype(int)

    row_order = ["A", "B", "C"]
    sub["row_index"] = sub["row"].map({r: i for i, r in enumerate(row_order)})

    fig, ax = plt.subplots(figsize=(6, 4))

    # =======================================================
    # COLORING RULES — NOW USING ONLY "group"
    # =======================================================
    if use_group or double_label:

        if "group" not in sub.columns:
            raise ValueError(
                "Column 'group' is missing. Did you run assign_group()?"
            )

        # fixed colors for 3 biological groups
        color_map = {
            "Mutant": "#FF9999",
            "WT":     "#99CCFF",
            "GM":     "#CCFFCC",
        }

        # ensure unknown groups get a color
        for g in sub["group"].unique():
            if g not in color_map:
                color_map[g] = "#DDDDDD"

    else:
        # default behavior: color wells by cell line
        cell_lines = sorted(sub["cell_line"].unique())
        color_map = {cl: plt.cm.Set3(i / len(cell_lines))
                     for i, cl in enumerate(cell_lines)}

    # =======================================================
    # DRAW WELLS
    # =======================================================
    for _, r in sub.iterrows():
        x = r["col"] - 1
        y = r["row_index"]

        # determine color
        if use_group or double_label:
            fill_key = r["group"]
        else:
            fill_key = r["cell_line"]

        rect = patches.Rectangle(
            (x, y),
            1, 1,
            linewidth=1,
            edgecolor='black',
            facecolor=color_map[fill_key]
        )
        ax.add_patch(rect)

        # =======================================================
        # LABEL RULES
        # =======================================================
        if double_label:
            # group on top
            ax.text(
                x + 0.5, y + 0.35,
                r["group"],
                ha='center', va='center',
                fontsize=12, fontweight='bold'
            )
            # cell line under it
            ax.text(
                x + 0.5, y + 0.65,
                f"({r['cell_line']})",
                ha='center', va='center',
                fontsize=10
            )

        elif use_group:
            ax.text(
                x + 0.5, y + 0.5,
                r["group"],
                ha='center', va='center',
                fontsize=12, fontweight='bold'
            )

        else:
            ax.text(
                x + 0.5, y + 0.5,
                r["cell_line"],
                ha='center', va='center',
                fontsize=12
            )

    # =======================================================
    # AXIS FORMATTING
    # =======================================================
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)

    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels([1, 2, 3, 4])

    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(row_order)

    if double_label:
        title = "Group + Cell Line Layout"
    elif use_group:
        title = "Group Layout"
    else:
        title = "Cell Line Layout"

    ax.set_title(f"{plate_name} — {title}")
    ax.invert_yaxis()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_feature_boxplots(df, feature, threshold_spikes_per_min=5, group_col="cell_line"):
    """
    Plot boxplots for a given feature across experiments.

    feature options:
        "firing"      -> wells + electrodes (threshold applies)
        "burst"       -> wells + electrodes (NO threshold)
        "network"     -> wells only
        "synchrony"   -> wells only
    """

    threshold_hz = threshold_spikes_per_min / 60.0

    feature = feature.lower()
    valid = ["firing", "burst", "network", "synchrony"]
    if feature not in valid:
        raise ValueError(f"feature must be one of {valid}")

    # well-level extraction
    if feature == "firing":
        df["well_value"] = df["firing_rate_stats_hz"].str[0]
    elif feature == "burst":
        df["well_value"] = df["burst_rate_stats_hz"].str[0]
    elif feature == "network":
        df["well_value"] = df["network_event_rate_hz"]
    elif feature == "synchrony":
        df["well_value"] = df["synchrony_index"]

    df["date_label"] = df["recording_date"].astype(str)

    for exp in sorted(df["experiment"].unique()):
        sub = df[df["experiment"] == exp].copy()
        print(f"\n=== Plotting Experiment {exp} ===")

        # WELL LEVEL
        plt.figure(figsize=(18,6))
        sns.boxplot(
            data=sub,
            x="date_label",
            y="well_value",
            hue=group_col,
            showfliers=False,
            width=0.7
        )
        plt.title(f"Experiment {exp} — Well-Level {feature.capitalize()} Rate")
        plt.xlabel("Recording date")
        plt.ylabel(f"{feature.capitalize()} rate (Hz)")
        plt.xticks(rotation=45)
        plt.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        # ELECTRODE LEVEL (firing and burst)
        if feature in ["firing", "burst"]:
            key = "firing_rate_vector_hz" if feature == "firing" else "burst_rate_vector_hz"

            def expand(row):
                return pd.DataFrame({
                    "plate_name": row["plate_name"],
                    "well": row["well"],
                    group_col: row[group_col],
                    "date_label": str(row["recording_date"]),
                    "value": row[key]
                })

            elec_df = pd.concat([expand(r) for _, r in sub.iterrows()], ignore_index=True)
            elec_df = elec_df.dropna(subset=["value"])

            if feature == "firing":
                elec_df = elec_df[elec_df["value"] >= threshold_hz]

            if elec_df.empty:
                print(f"No valid electrodes for {feature} in {exp}")
                continue

            plt.figure(figsize=(18,6))
            sns.boxplot(
                data=elec_df,
                x="date_label",
                y="value",
                hue=group_col,
                showfliers=False,
                width=0.7
            )
            plt.title(
                f"Experiment {exp} — Electrode-Level {feature.capitalize()} Rate"
                + (" (≥5 spikes/min)" if feature == "firing" else "")
            )
            plt.xlabel("Recording date")
            plt.ylabel(f"{feature.capitalize()} rate (Hz)")
            plt.xticks(rotation=45)
            plt.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.show()

def run_feature_stats(df, feature, threshold_spikes_per_min=5, group_col="cell_line"):
    """
    Runs ANOVAs and posthoc t-tests for ONE selected experiment.
    df should already be filtered to a single experiment.

    group_col = "cell_line" OR "group"
    """

    df = df.copy()
    feature = feature.lower()
    threshold_hz = threshold_spikes_per_min / 60.0

    # --- extract well-level values ---
    if feature == "firing":
        df["well_value"] = df["firing_rate_stats_hz"].str[0]
    elif feature == "burst":
        df["well_value"] = df["burst_rate_stats_hz"].str[0]
    elif feature == "network":
        df["well_value"] = df["network_event_rate_hz"]
    elif feature == "synchrony":
        df["well_value"] = df["synchrony_index"]

    df["date_label"] = df["recording_date"].astype(str)

    # -------------------
    # ANALYZE THIS EXPERIMENT ONLY
    # -------------------
    sub = df.dropna(subset=["well_value", group_col]).copy()
    exp_result = {"well": {}, "electrode": {}}

    # ---------------- WELL: two-way ANOVA ----------------
    if len(sub[group_col].unique()) > 1 and len(sub["date_label"].unique()) > 1:
        model = smf.ols(f"well_value ~ C({group_col}) * C(date_label)", data=sub).fit()
        exp_result["well"]["two_way_anova"] = _clean_anova_table(anova_lm(model, typ=2))
    else:
        exp_result["well"]["two_way_anova"] = None

    # ---------------- WELL: one-way ANOVA ----------------
    if len(sub[group_col].unique()) > 1:
        model = smf.ols(f"well_value ~ C({group_col})", data=sub).fit()
        exp_result["well"]["one_way_anova"] = _clean_anova_table(anova_lm(model, typ=2))
    else:
        exp_result["well"]["one_way_anova"] = None

    # ---------------- WELL: posthoc t-tests ----------------
    from itertools import combinations
    groups = sorted(sub[group_col].unique())
    pairs = list(combinations(groups, 2))
    m = len(pairs)
    posthoc_rows = []

    for a, b in pairs:
        xa = sub.loc[sub[group_col] == a, "well_value"].dropna()
        xb = sub.loc[sub[group_col] == b, "well_value"].dropna()
        if len(xa) < 2 or len(xb) < 2:
            continue
        t, p = ttest_ind(xa, xb, equal_var=False)
        bonf = min(p * m, 1.0)
        posthoc_rows.append({
            "group1": a,
            "group2": b,
            "mean1": float(xa.mean()),
            "mean2": float(xb.mean()),
            "difference": float(xa.mean() - xb.mean()),
            "t": float(t),
            "p_raw": float(p),
            "p_fmt": _format_p(p),
            "p_bonf": float(bonf),
            "p_bonf_fmt": _format_p(bonf)
        })

    exp_result["well"]["posthoc"] = pd.DataFrame(posthoc_rows) if posthoc_rows else None

    # ---------------- ELECTRODE LEVEL ----------------
    if feature in ["firing", "burst"]:
        key = "firing_rate_vector_hz" if feature == "firing" else "burst_rate_vector_hz"

        expanded = []
        for _, row in sub.iterrows():
            for v in row[key]:
                expanded.append({
                    group_col: row[group_col],
                    "date_label": row["date_label"],
                    "value": v
                })

        elec_df = pd.DataFrame(expanded).dropna(subset=["value"])

        if feature == "firing":
            elec_df = elec_df[elec_df["value"] >= threshold_hz]

        # 2-way ANOVA electrode
        if len(elec_df[group_col].unique()) > 1 and len(elec_df["date_label"].unique()) > 1:
            model = smf.ols(f"value ~ C({group_col}) * C(date_label)", data=elec_df).fit()
            exp_result["electrode"]["two_way_anova"] = _clean_anova_table(anova_lm(model, typ=2))
        else:
            exp_result["electrode"]["two_way_anova"] = None

        # 1-way ANOVA electrode
        if len(elec_df[group_col].unique()) > 1:
            model = smf.ols(f"value ~ C({group_col})", data=elec_df).fit()
            exp_result["electrode"]["one_way_anova"] = _clean_anova_table(anova_lm(model, typ=2))
        else:
            exp_result["electrode"]["one_way_anova"] = None

    return exp_result


def _format_p(p, ndigits=6):
    """Format p-values as decimals with fixed number of digits."""
    if p is None or np.isnan(p):
        return ""
    return f"{p:.{ndigits}f}"


def _clean_anova_table(anova):
    """Rename columns, add p-formatted column."""
    cleaned = anova.rename(
        columns={
            "sum_sq": "SS",
            "df": "DF",
            "F": "F_statistic",
            "PR(>F)": "p_value"
        }
    )
    cleaned["p_formatted"] = cleaned["p_value"].apply(_format_p)
    return cleaned



def assign_group(line):
    mutant_lines = {"C3", "C71"}
    wt_lines = { "C30"}
    other_control = {"H9", "C90", "C12","C54"}
    if line in mutant_lines:
        return "Mutant"
    elif line in wt_lines:
        return "WT"
    elif line in other_control:
        return "other"
    else:
        return "GM"


# --------------------------------------------------------------
# MASTER FUNCTION FOR FULL MEA ANALYSIS
# --------------------------------------------------------------
def run_full_mea_analysis(
    df,
    experiment,
    assign_groups=True,
    use_group_for_plate_map=False,
    double_label=False,
    threshold_spikes_per_min=5
):
    """
    Full MEA analysis pipeline:
      • filters dataframe by experiment
      • assigns Mutant / WT / GM group (if requested)
      • draws plate maps
      • creates boxplots for cell lines
      • creates boxplots for groups (Mutant / WT / GM)
      • runs stats for cell lines + prints posthoc
      • runs stats for groups + prints posthoc
      • returns full stats dictionary
    """

    # ---------------- FILTER BY EXPERIMENT ----------------
    df_exp = df[df["experiment"] == experiment].copy()
    if df_exp.empty:
        raise ValueError(
            f"No rows found for experiment '{experiment}'. "
            f"Available experiments: {df['experiment'].unique()}"
        )

    print(f"\n### RUNNING ANALYSIS FOR {experiment} ###")
    print("Plates included:", df_exp["plate_name"].unique())

    # ---------------- ASSIGN GROUPS ----------------
    if assign_groups:
        df_exp["group"] = df_exp["cell_line"].apply(assign_group)

    # ---------------- PLOT PLATE MAPS ----------------
    print("\n### DRAWING PLATE MAPS ###")
    for plate in df_exp["plate_name"].unique():
        print(f" → Plotting {plate}")
        plot_plate_map(
            df_exp,
            plate_name=plate,
            use_group=use_group_for_plate_map,
            double_label=double_label
        )

    # ---------------- FEATURES ----------------
    features = ["firing", "burst", "network", "synchrony"]

    # -----------------------------------------------------------
    # 1) CELL-LINE PLOTS
    # -----------------------------------------------------------
    print("\n### GENERATING FEATURE PLOTS — CELL LINES ###")
    for feat in features:
        print(f" → Plotting {feat} (cell lines)")
        plot_feature_boxplots(
            df_exp,
            feature=feat,
            threshold_spikes_per_min=threshold_spikes_per_min,
            group_col="cell_line"   # explicit for clarity
        )

    # -----------------------------------------------------------
    # 2) GROUP PLOTS (Mutant / WT / GM)
    # -----------------------------------------------------------
    print("\n### GENERATING FEATURE PLOTS — GROUPS ###")

    df_group = df_exp.copy()  # KEEP both group and cell_line intact

    for feat in features:
        print(f" → Plotting {feat} (groups)")
        plot_feature_boxplots(
            df_group,
            feature=feat,
            threshold_spikes_per_min=threshold_spikes_per_min,
            group_col="group"     # key fix
        )

    # -----------------------------------------------------------
    # 3) STATS — CELL LINE
    # -----------------------------------------------------------
    print("\n### RUNNING STATISTICS — CELL LINES ###")
    stats_cell = {}

    for feat in features:
        print(f"\n===== CELL LINE STATS: {feat.upper()} =====")
        res = run_feature_stats(
            df_exp,
            feature=feat,
            threshold_spikes_per_min=threshold_spikes_per_min,
            group_col="cell_line"
        )
        stats_cell[feat] = res

        # ---- PRINT POSTHOC TABLES ----
        if res["well"]["posthoc"] is not None:
            print("\n  WELL-LEVEL POST-HOC t-tests:")
            print(res["well"]["posthoc"])

        if res.get("electrode", {}).get("posthoc") is not None:
            print("\n  ELECTRODE-LEVEL POST-HOC t-tests:")
            print(res["electrode"]["posthoc"])

    # -----------------------------------------------------------
    # 4) STATS — GROUP LEVEL
    # -----------------------------------------------------------
    print("\n### RUNNING STATISTICS — GROUPS (Mutant / WT / GM) ###")
    stats_group = {}

    df_group_stats = df_exp.copy()

    for feat in features:
        print(f"\n===== GROUP STATS: {feat.upper()} =====")
        res_g = run_feature_stats(
            df_group_stats,
            feature=feat,
            threshold_spikes_per_min=threshold_spikes_per_min,
            group_col="group"
        )
        stats_group[feat] = res_g

        # ---- PRINT POSTHOC TABLES ----
        if res_g["well"]["posthoc"] is not None:
            print("\n  WELL-LEVEL POST-HOC t-tests (groups):")
            print(res_g["well"]["posthoc"])

        if res_g.get("electrode", {}).get("posthoc") is not None:
            print("\n  ELECTRODE-LEVEL POST-HOC t-tests (groups):")
            print(res_g["electrode"]["posthoc"])

    print("\n### ANALYSIS COMPLETE ###")

    return {
        "experiment": experiment,
        "stats_cell_line": stats_cell,
        "stats_group": stats_group,
    }

def remove_clones_by_plate(df, removals):
    """
    Removes unwanted clones based ONLY on plate_name.
    Applies the same removal rules across all experiments and dates.
    """


    df_clean = df.copy()
    mask_remove = []

    for idx, row in df_clean.iterrows():
        plate = row["plate_name"]
        clone = row["cell_line"]

        # If plate has removal rules
        if plate in removals:
            if clone in removals[plate]:
                mask_remove.append(True)
                continue

        mask_remove.append(False)

    mask_remove = pd.Series(mask_remove, index=df_clean.index)

    print(f"Removed {mask_remove.sum()} rows based on plate-specific clone removal.")

    return df_clean[~mask_remove].reset_index(drop=True)
