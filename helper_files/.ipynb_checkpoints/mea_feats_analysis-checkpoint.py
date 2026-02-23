import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as mc
from scipy.stats import linregress, ttest_ind



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
            "Ctrl":     "#99CCFF",
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
    mutant_lines = {"C90", "C90m"}
    ctrl_lines = { "C30", "C12c"}
    other = {"H9", "C90", "C12","C54", "C3", "C71"}
    if line in mutant_lines:
        return "Mutant"
    elif line in ctrl_lines:
        return "Ctrl"
    elif line in other:
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

# --------------------------------------------------------------
# FUNCTIONS FOR FIGURE 
# --------------------------------------------------------------
# Updated Palette with LaTeX superscript for the GM line
MAOA_FIGURE_PALETTE = {
    "Xa:WT/Xi:MUT": "#9C899C", 
    "Xa:MUT/Xi:WT": "#EAC3EB", 
    "MAOA/B$^{-/Y}$": "#E980C0"  
}

PLOT_ORDER = ["Xa:WT/Xi:MUT", "Xa:MUT/Xi:WT", "MAOA/B$^{-/Y}$"]


def get_stars(p):
    if p is None or np.isnan(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."

def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0

def run_comprehensive_stats(df, feature, level="well", unit="months"):
    """
    Handles both 'well' and 'electrode' level stats. 
    Returns ANOVA (Interaction) and a detailed comparison table.
    """
    df = df.copy()
    
    # Pre-process for level
    if level == "electrode":
        key = "firing_rate_vector_hz" if feature == "firing" else "burst_rate_vector_hz"
        expanded = []
        for _, row in df.iterrows():
            for v in row[key]:
                expanded.append({"group": row["group"], "time_val": row["time_val"], "val": v})
        df_stats = pd.DataFrame(expanded).dropna()
        if feature == "firing": df_stats = df_stats[df_stats["val"] >= (5/60)]
    else:
        if feature == "firing": df_stats = df.assign(val=df["firing_rate_stats_hz"].str[0])
        elif feature == "burst": df_stats = df.assign(val=df["burst_rate_stats_hz"].str[0])
        elif feature == "network": df_stats = df.assign(val=df["network_event_rate_hz"])
        elif feature == "synchrony": df_stats = df.assign(val=df["synchrony_index"])
        df_stats = df_stats.dropna(subset=["val"])

    # 1. Trajectory Interaction (ANOVA)
    model = smf.ols("val ~ C(group) * C(time_val)", data=df_stats).fit()
    anova_res = anova_lm(model, typ=2)
    
    # 2. Time-Point Table
    timepoint_results = []
    for time in sorted(df_stats["time_val"].unique()):
        sub = df_stats[df_stats["time_val"] == time]
        if len(sub["group"].unique()) > 1:
            m_pt = smf.ols("val ~ C(group)", data=sub).fit()
            p_anova = anova_lm(m_pt, typ=2).loc["C(group)", "PR(>F)"]
            res = {"Time": time, "ANOVA_p": p_anova}
            if p_anova < 0.05:
                ph = mc.MultiComparison(sub["val"], sub["group"]).tukeyhsd()
                ph_df = pd.DataFrame(data=ph.summary().data[1:], columns=ph.summary().data[0])
                for _, row in ph_df.iterrows():
                    res[f"{row['group1']} vs {row['group2']}"] = row['p-adj']
            timepoint_results.append(res)
            
    return {"anova": anova_res, "table": pd.DataFrame(timepoint_results)}


def plot_figure_boxplots_for_paper(df, feature, palette=MAOA_FIGURE_PALETTE, order=PLOT_ORDER, unit="months", threshold_spikes_per_min=5):
    df = df.copy()
    threshold_hz = threshold_spikes_per_min / 60.0
    
    # 1. Feature Mapping logic
    elec_key = None
    if feature == "firing":
        df["plot_val"] = df["firing_rate_stats_hz"].str[0]
        ylabel_base = "Firing Rate"
        elec_key = "firing_rate_vector_hz"
    elif feature == "burst":
        df["plot_val"] = df["burst_rate_stats_hz"].str[0]
        ylabel_base = "Burst Rate"
        elec_key = "burst_rate_vector_hz"
    elif feature == "network":
        df["plot_val"] = df["network_event_rate_hz"]
        ylabel_base = "Network Event Rate"
    elif feature == "synchrony":
        df["plot_val"] = df["synchrony_index"]
        ylabel_base = "Synchrony Index"

    xlabel = f"Age ({unit.capitalize()})"

    def draw_trajectory_panel(data, y_col, level_name):
        data = data.dropna(subset=[y_col]).sort_values("age_days")
        if data.empty: return

        # Trajectory Stats
        try:
            model = smf.ols(f"{y_col} ~ C(group) * C(time_val)", data=data).fit()
            p_int = anova_lm(model, typ=2).loc["C(group):C(time_val)", "PR(>F)"]
            traj_summary = f"Trajectory: {get_stars(p_int)}"
            
            posthoc_text = ""
            if p_int < 0.05:
                ph = mc.MultiComparison(data[y_col], data["group"]).tukeyhsd()
                ph_df = pd.DataFrame(data=ph.summary().data[1:], columns=ph.summary().data[0])
                sigs = [f"{r['group1']} v {r['group2']}: {get_stars(r['p-adj'])}" for _, r in ph_df.iterrows() if r['p-adj'] < 0.05]
                posthoc_text = "\n".join(sigs)
        except:
            traj_summary, posthoc_text = "Trajectory: N/A", ""

        plt.figure(figsize=(12, 7))
        sns.set_style("ticks")
        ax = sns.boxplot(data=data, x="time_val", y=y_col, hue="group", hue_order=order, 
                         palette=palette, showfliers=False, width=0.6, 
                         boxprops=dict(alpha=0.8, edgecolor='black'))
        
        sns.stripplot(data=data, x="time_val", y=y_col, hue="group", hue_order=order, 
                      palette=palette, dodge=True, alpha=0.55, jitter=0.2, size=3, legend=False)

        # Regression Lines
        time_points = sorted(data["time_val"].unique())
        x_map = {val: i for i, val in enumerate(time_points)}
        data['x_idx'] = data['time_val'].map(x_map)
        for g in order:
            sub = data[data["group"] == g]
            if len(sub) > 2:
                sns.regplot(data=sub, x="x_idx", y=y_col, scatter=False, color=palette[g], 
                            truncate=True, line_kws={"ls":"--", "lw":2.5})

        plt.ylabel(f"{level_name} {ylabel_base} (Hz)", fontsize=12)
        plt.xlabel(xlabel, fontsize=12)
        plt.text(0.97, 0.95, f"{traj_summary}\n{posthoc_text}".strip(), transform=ax.transAxes, 
                 fontsize=9, fontweight='bold', ha='right', va='top', 
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))
        
        plt.legend(title="Genotype", bbox_to_anchor=(1.05, 0.7), loc='upper left', frameon=False)
        sns.despine(); plt.tight_layout(); plt.show()

    # Draw Well-Level
    draw_trajectory_panel(df, "plot_val", "Well-Level")

    # Draw Electrode-Level (if applicable)
    if elec_key and elec_key in df.columns:
        expanded = []
        for _, row in df.iterrows():
            if isinstance(row[elec_key], (list, np.ndarray)):
                for v in row[elec_key]:
                    expanded.append({"group": row["group"], "time_val": row["time_val"], "age_days": row["age_days"], "v": v})
        
        if expanded:
            elec_df = pd.DataFrame(expanded).dropna()
            if feature == "firing": elec_df = elec_df[elec_df["v"] >= threshold_hz]
            draw_trajectory_panel(elec_df, "v", "Electrode-Level")

#GROUPED 
def plot_grouped_genotype_analysis(df, feature, palette, order, threshold_spikes_per_min=5):
    df = df.copy()
    threshold_hz = threshold_spikes_per_min / 60.0
    FONT_SIZE = 12

    def draw_grouped_panel(data, y_col, level_name, feat_label):
        data = data.dropna(subset=[y_col])
        if data.empty: return
        
        try:
            model = smf.ols(f"{y_col} ~ C(group)", data=data).fit()
            p_anova = anova_lm(model, typ=2).loc["C(group)", "PR(>F)"]
            anova_summary = f"ANOVA: {get_stars(p_anova)}"
            comp = mc.MultiComparison(data[y_col], data["group"])
            ph_df = pd.DataFrame(data=comp.tukeyhsd().summary().data[1:], columns=comp.tukeyhsd().summary().data[0])
            posthoc_data = [{"g1": r['group1'], "g2": r['group2'], "star": get_stars(r['p-adj']), 
                             "d": calculate_cohens_d(data[data["group"]==r['group1']][y_col], 
                                                     data[data["group"]==r['group2']][y_col])} for _, r in ph_df.iterrows()]
        except: 
            anova_summary, posthoc_data = "ANOVA Error", []

        plt.figure(figsize=(14, 8))
        sns.set_style("ticks")
        ax = sns.boxplot(data=data, x="group", y=y_col, order=order, palette=palette, showfliers=False, width=0.5, boxprops=dict(alpha=0.8, edgecolor='black'))
        sns.stripplot(data=data, x="group", y=y_col, order=order, palette=palette, alpha=0.25, jitter=0.2, size=5, legend=False)

        res_x, res_y = 1.05, 0.90
        plt.text(res_x, 0.96, anova_summary, transform=ax.transAxes, fontsize=FONT_SIZE, fontweight='bold', va='top')
        curr_y = res_y
        for res in posthoc_data:
            ax.text(res_x, curr_y, f" {res['g1']} ", transform=ax.transAxes, fontweight='bold', fontsize=FONT_SIZE, va='top', bbox=dict(boxstyle='round,pad=0.2', facecolor=palette[res['g1']], alpha=0.6, edgecolor='none'))
            ax.text(res_x + 0.38, curr_y, f" {res['g2']} ", transform=ax.transAxes, fontweight='bold', fontsize=FONT_SIZE, va='top', bbox=dict(boxstyle='round,pad=0.2', facecolor=palette[res['g2']], alpha=0.6, edgecolor='none'))
            ax.text(res_x + 0.33, curr_y, "vs", transform=ax.transAxes, fontsize=FONT_SIZE, va='top', ha='right')
            ax.text(res_x, curr_y - 0.05, f"({res['star']}, Cohen's d: {res['d']:.2f})", transform=ax.transAxes, fontsize=FONT_SIZE, va='top')
            curr_y -= 0.14 

        plt.title(f"Grouped {level_name} {feat_label}", fontsize=FONT_SIZE + 2, fontweight='bold')
        plt.ylabel(f"{level_name} {feat_label} (Hz)" if "Index" not in feat_label else f"{level_name} {feat_label}", fontsize=FONT_SIZE + 2)
        plt.xlabel("Genotype", fontsize=FONT_SIZE)
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
        sns.despine()
        plt.subplots_adjust(right=0.55); plt.show()

    # Feature Logic
    if feature == "firing":
        df["well_val"] = df["firing_rate_stats_hz"].str[0]
        draw_grouped_panel(df, "well_val", "Well-Level", "Firing Rate")
        expanded = [{"group": r["group"], "v": v} for _, r in df.iterrows() for v in r["firing_rate_vector_hz"]]
        e_df = pd.DataFrame(expanded).dropna()
        e_df = e_df[e_df["v"] >= threshold_hz]
        draw_grouped_panel(e_df, "v", "Electrode-Level", "Firing Rate")
        
    elif feature == "burst":
        df["well_val"] = df["burst_rate_stats_hz"].str[0]
        draw_grouped_panel(df, "well_val", "Well-Level", "Burst Rate")
        expanded = [{"group": r["group"], "v": v} for _, r in df.iterrows() for v in r["burst_rate_vector_hz"]]
        e_df = pd.DataFrame(expanded).dropna()
        draw_grouped_panel(e_df, "v", "Electrode-Level", "Burst Rate")
        
    elif feature == "network":
        df["well_val"] = df["network_event_rate_hz"]
        draw_grouped_panel(df, "well_val", "Well-Level", "Network Rate")
        
    elif feature == "synchrony":
        df["well_val"] = df["synchrony_index"]
        draw_grouped_panel(df, "well_val", "Well-Level", "Synchrony Index")


def run_mea_analysis_for_figure(df, experiment, unit="days", palette=MAOA_FIGURE_PALETTE, threshold_spikes_per_min=5):
    df_exp = df[df["experiment"] == experiment].copy()
    
    def assign_genotype(line):
        if line in {"C30", "C12c"}: return "Xa:WT/Xi:MUT"
        if line in {"C90", "C90m"}: return "Xa:MUT/Xi:WT"
        if line in {"GM", "MAOA/B−/Y"}: return "MAOA/B$^{-/Y}$"
        return "other"
    
    df_exp["group"] = df_exp["cell_line"].apply(assign_genotype)
    
    # Age Calculation
    ref_date = pd.to_datetime("2025-08-12")
    df_exp["recording_date"] = pd.to_datetime(df_exp["recording_date"])
    df_exp["age_days"] = (df_exp["recording_date"] - ref_date).dt.days + 180
    df_exp["time_val"] = (df_exp["age_days"] / 30.44).round(1) if unit == "months" else df_exp["age_days"]

    for feat in ["firing", "burst", "network", "synchrony"]:
        print(f"\n{'='*40}\nFEATURE: {feat.upper()}\n{'='*40}")
        plot_figure_boxplots_for_paper(df_exp, feat, palette, PLOT_ORDER, unit)
        
        # Well-Level Stats Table
        print("\n--- WELL-LEVEL STATS ---")
        well_stats = run_comprehensive_stats(df_exp, feat, level="well", unit=unit)
        print("ANOVA Interaction:\n", well_stats["anova"])
        print("Time-Point Table:\n", well_stats["table"].to_string(index=False))
        
        # Electrode-Level Stats Table (Only for Firing/Burst)
        if feat in ["firing", "burst"]:
            print(f"\n--- ELECTRODE-LEVEL {feat.upper()} STATS ---")
            elec_stats = run_comprehensive_stats(df_exp, feat, level="electrode", unit=unit)
            print("ANOVA Interaction:\n", elec_stats["anova"])
            print("Time-Point Table:\n", elec_stats["table"].to_string(index=False))

# For Grouped Analysis (Separate Run)
def run_grouped_analysis(df, experiment):
    df_exp = df[df["experiment"] == experiment].copy()
    
    # 1. ACTUAL Genotype assignment logic
    def assign_genotype(line):
        if line in {"C30", "C12c"}: return "Xa:WT/Xi:MUT"
        if line in {"C90", "C90m"}: return "Xa:MUT/Xi:WT"
        if line in {"GM", "MAOA/B−/Y"}: return "MAOA/B$^{-/Y}$"
        return "other"
    
    df_exp["group"] = df_exp["cell_line"].apply(assign_genotype)

    # 2. Filter out 'other' to keep the plot clean
    df_exp = df_exp[df_exp["group"] != "other"]

    # 3. Run the analysis
    for feat in ["firing", "burst", "network", "synchrony"]:
        print(f"\n--- GENERATING GROUPED PLOTS FOR {feat.upper()} ---")
        plot_grouped_genotype_analysis(df_exp, feat, MAOA_FIGURE_PALETTE, PLOT_ORDER)