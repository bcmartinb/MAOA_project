import os
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

from helper_files.maoa_plate_loader import *

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
    """
    sub = df[df["plate_name"] == plate_name].copy()
    if sub.empty:
        print(f"No data found for plate {plate_name}")
        return

    sub = sub.drop_duplicates(subset=["well"]).copy()

    sub["row"] = sub["well"].str[0]
    sub["col"] = sub["well"].str[1:].astype(int)

    row_order = ["A", "B", "C"]
    sub["row_index"] = sub["row"].map({r: i for i, r in enumerate(row_order)})

    fig, ax = plt.subplots(figsize=(6, 4))

    if use_group or double_label:
        if "group" not in sub.columns:
            raise ValueError(
                "Column 'group' is missing. Did you run assign_group()?"
            )

        color_map = {
            "Mutant": "#EAC3EB",
            "Ctrl":     "#9C899C",
            "GM":     "#E980C0",
        }

        for g in sub["group"].unique():
            if g not in color_map:
                color_map[g] = "#DDDDDD"
    else:
        cell_lines = sorted(sub["cell_line"].unique())
        color_map = {cl: plt.cm.Set3(i / len(cell_lines))
                     for i, cl in enumerate(cell_lines)}

    for _, r in sub.iterrows():
        x = r["col"] - 1
        y = r["row_index"]

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

        if double_label:
            ax.text(
                x + 0.5, y + 0.35,
                r["group"],
                ha='center', va='center',
                fontsize=12, fontweight='bold'
            )
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
    """
    threshold_hz = threshold_spikes_per_min / 60.0

    feature = feature.lower()
    valid = ["firing", "burst", "network", "synchrony"]
    if feature not in valid:
        raise ValueError(f"feature must be one of {valid}")

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

        plt.figure(figsize=(18,6))
        sns.stripplot(data=sub, x="date_label", y="well_value", hue=group_col,
                      color='#888888', dodge=True, alpha=0.6, jitter=0.2, size=5,
                      edgecolor='none', legend=False, zorder=2)
        sns.boxplot(
            data=sub,
            x="date_label",
            y="well_value",
            hue=group_col,
            showfliers=False,
            width=0.7,
            boxprops=dict(alpha=0.8),
            zorder=3,
        )
        plt.title(f"Experiment {exp} — Well-Level {feature.capitalize()} Rate")
        plt.xlabel("Recording date")
        plt.ylabel(f"{feature.capitalize()} rate (Hz)")
        plt.xticks(rotation=45)
        plt.legend(title=group_col, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

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
            sns.stripplot(data=elec_df, x="date_label", y="value", hue=group_col,
                          color='#888888', dodge=True, alpha=0.6, jitter=0.2, size=5,
                          edgecolor='none', legend=False, zorder=2)
            sns.boxplot(
                data=elec_df,
                x="date_label",
                y="value",
                hue=group_col,
                showfliers=False,
                width=0.7,
                boxprops=dict(alpha=0.8),
                zorder=3,
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
    df = df.copy()
    feature = feature.lower()
    threshold_hz = threshold_spikes_per_min / 60.0

    if feature == "firing":
        df["well_value"] = df["firing_rate_stats_hz"].str[0]
    elif feature == "burst":
        df["well_value"] = df["burst_rate_stats_hz"].str[0]
    elif feature == "network":
        df["well_value"] = df["network_event_rate_hz"]
    elif feature == "synchrony":
        df["well_value"] = df["synchrony_index"]

    df["date_label"] = df["recording_date"].astype(str)

    sub = df.dropna(subset=["well_value", group_col]).copy()
    exp_result = {"well": {}, "electrode": {}}

    if len(sub[group_col].unique()) > 1 and len(sub["date_label"].unique()) > 1:
        model = smf.ols(f"well_value ~ C({group_col}) * C(date_label)", data=sub).fit()
        exp_result["well"]["two_way_anova"] = _clean_anova_table(anova_lm(model, typ=2))
    else:
        exp_result["well"]["two_way_anova"] = None

    if len(sub[group_col].unique()) > 1:
        model = smf.ols(f"well_value ~ C({group_col})", data=sub).fit()
        exp_result["well"]["one_way_anova"] = _clean_anova_table(anova_lm(model, typ=2))
    else:
        exp_result["well"]["one_way_anova"] = None

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

        if len(elec_df[group_col].unique()) > 1 and len(elec_df["date_label"].unique()) > 1:
            model = smf.ols(f"value ~ C({group_col}) * C(date_label)", data=elec_df).fit()
            exp_result["electrode"]["two_way_anova"] = _clean_anova_table(anova_lm(model, typ=2))
        else:
            exp_result["electrode"]["two_way_anova"] = None

        if len(elec_df[group_col].unique()) > 1:
            model = smf.ols(f"value ~ C({group_col})", data=elec_df).fit()
            exp_result["electrode"]["one_way_anova"] = _clean_anova_table(anova_lm(model, typ=2))
        else:
            exp_result["electrode"]["one_way_anova"] = None

    return exp_result

def _format_p(p, ndigits=6):
    if p is None or np.isnan(p):
        return ""
    return f"{p:.{ndigits}f}"

def _clean_anova_table(anova):
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
    mutant_lines = {"C71", "C3"}
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

def run_full_mea_analysis(
    df,
    experiment,
    assign_groups=True,
    use_group_for_plate_map=False,
    double_label=False,
    threshold_spikes_per_min=5
):
    df_exp = df[df["experiment"] == experiment].copy()
    if df_exp.empty:
        raise ValueError(
            f"No rows found for experiment '{experiment}'. "
            f"Available experiments: {df['experiment'].unique()}"
        )

    print(f"\n### RUNNING ANALYSIS FOR {experiment} ###")
    print("Plates included:", df_exp["plate_name"].unique())

    if assign_groups:
        df_exp["group"] = df_exp["cell_line"].apply(assign_group)

    print("\n### DRAWING PLATE MAPS ###")
    for plate in df_exp["plate_name"].unique():
        print(f" → Plotting {plate}")
        plot_plate_map(
            df_exp,
            plate_name=plate,
            use_group=use_group_for_plate_map,
            double_label=double_label
        )

    features = ["firing", "burst", "network", "synchrony"]

    print("\n### GENERATING FEATURE PLOTS — CELL LINES ###")
    for feat in features:
        print(f" → Plotting {feat} (cell lines)")
        plot_feature_boxplots(
            df_exp,
            feature=feat,
            threshold_spikes_per_min=threshold_spikes_per_min,
            group_col="cell_line"
        )

    print("\n### GENERATING FEATURE PLOTS — GROUPS ###")
    df_group = df_exp.copy()

    for feat in features:
        print(f" → Plotting {feat} (groups)")
        plot_feature_boxplots(
            df_group,
            feature=feat,
            threshold_spikes_per_min=threshold_spikes_per_min,
            group_col="group"
        )

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

        if res["well"]["posthoc"] is not None:
            print("\n  WELL-LEVEL POST-HOC t-tests:")
            print(res["well"]["posthoc"])

        if res.get("electrode", {}).get("posthoc") is not None:
            print("\n  ELECTRODE-LEVEL POST-HOC t-tests:")
            print(res["electrode"]["posthoc"])

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
    df_clean = df.copy()
    mask_remove = []

    for idx, row in df_clean.iterrows():
        plate = row["plate_name"]
        clone = row["cell_line"]

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

MAOA_FIGURE_PALETTE = {
    "Xa:WT/Xi:MUT": "#8c778c", 
    "Xa:MUT/Xi:WT": "#e7b3e8", 
    "MAOA/B$^{-/Y}$": "#e66ab6"  
}

PLOT_ORDER = ["Xa:WT/Xi:MUT", "Xa:MUT/Xi:WT", "MAOA/B$^{-/Y}$"]

def get_stars(p):
    if p is None or np.isnan(p): return "ns"
    if p < 0.0001: return "****"
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0

def run_comprehensive_stats(df, feature, level="well", unit="months"):
    df = df.copy()
    
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

    model = smf.ols("val ~ C(group) * C(time_val)", data=df_stats).fit()
    anova_res = anova_lm(model, typ=2)
    
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


def plot_figure_boxplots_for_paper(df, feature, palette=MAOA_FIGURE_PALETTE, order=PLOT_ORDER, unit="months", threshold_spikes_per_min=5, save_dir=None):
    df = df.copy()
    threshold_hz = threshold_spikes_per_min / 60.0
    
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

    def draw_trajectory_panel(data, y_col, level_name, split_axis=False, zoom_max_override=None, broken_axis=False):
        data = data.dropna(subset=[y_col]).sort_values("age_days")
        if data.empty: return

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

        # ── helper: draw strips + boxes + regression on a given axis ──────────
        def _plot_on(ax):
            ax.yaxis.grid(False)
            ax.xaxis.grid(False)
            ax.set_axisbelow(True)
            sns.stripplot(data=data, x="time_val", y=y_col, hue="group",
                          hue_order=order, color='#888888', dodge=True,
                          alpha=0.7, jitter=0.25, size=6,
                          edgecolor='none', legend=False, zorder=2, ax=ax)
            sns.boxplot(data=data, x="time_val", y=y_col, hue="group",
                        hue_order=order, palette=palette, showfliers=False,
                        width=0.6,
                        boxprops=dict(alpha=0.8, edgecolor='black', linewidth=1),
                        medianprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1),
                        whiskerprops=dict(color='black', linewidth=1),
                        zorder=3, ax=ax)
            time_points = sorted(data["time_val"].unique())
            x_map = {val: i for i, val in enumerate(time_points)}
            d2 = data.copy()
            d2['x_idx'] = d2['time_val'].map(x_map)
            for g in order:
                sub = d2[d2["group"] == g]
                if len(sub) > 2:
                    sns.regplot(data=sub, x="x_idx", y=y_col, scatter=False,
                                color=palette[g], truncate=True,
                                line_kws={"ls": "--", "lw": 2.5, "zorder": 4},
                                ax=ax)
            if ax.get_legend():
                ax.get_legend().remove()

        if split_axis:
            # zoom_max: include every whisker top but clip outlier dots
            whisker_tops = []
            for g in data['group'].unique():
                for t in data['time_val'].unique():
                    sub = (data[(data['group'] == g) & (data['time_val'] == t)]
                           [y_col].dropna())
                    if len(sub) >= 2:
                        q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
                        fence = q3 + 1.5 * (q3 - q1)
                        non_out = sub[sub <= fence]
                        whisker_tops.append(non_out.max() if not non_out.empty else q3)
            zoom_max = zoom_max_override if zoom_max_override is not None else (
                (max(whisker_tops) if whisker_tops
                 else data[y_col].quantile(0.9)) * 1.10)
            full_max = data[y_col].max() * 1.12

            # ── Figure layout: main (zoomed) left, inset (full) right ─────────
            fig = plt.figure(figsize=(14, 8))
            sns.set_style("white")

            # Main axes — left 55 % of figure
            ax = fig.add_axes([0.09, 0.14, 0.52, 0.80])
            _plot_on(ax)
            ax.set_ylim(0, zoom_max)
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(f"{level_name} {ylabel_base} (Hz)",
                          fontsize=30, color='black', fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=30, color='black', fontweight='bold')
            ax.tick_params(axis='both', colors='black', length=7,
                           width=1.5, labelsize=26)
            ax.text(0.03, 0.97,
                    f"{traj_summary}\n{posthoc_text}".strip(),
                    transform=ax.transAxes,
                    fontsize=20, fontweight='bold', ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

            # Inset axes — right 28 % of figure, full y range
            ax_inset = fig.add_axes([0.66, 0.14, 0.28, 0.80])
            _plot_on(ax_inset)
            ax_inset.set_ylim(0, full_max)
            ax_inset.spines['top'].set_visible(False)
            ax_inset.spines['right'].set_visible(False)
            ax_inset.spines['bottom'].set_color('black')
            ax_inset.spines['left'].set_color('black')
            ax_inset.tick_params(axis='both', colors='black', length=7,
                                 width=1.5, labelsize=26)
            ax_inset.set_xlabel("")
            ax_inset.set_ylabel("")
            ax_inset.set_xticklabels([])

        elif broken_axis:
            # ── Broken y-axis: bottom panel zoomed, top panel outliers ─────────
            # Use a global fence across all data to find the highest whisker top,
            # then add 25% padding so seaborn's per-box whiskers always fit.
            vals = data[y_col].dropna()
            q1_g, q3_g = vals.quantile(0.25), vals.quantile(0.75)
            global_fence = q3_g + 1.5 * (q3_g - q1_g)
            global_whisker_top = vals[vals <= global_fence].max()
            auto_zoom = global_whisker_top * 1.25
            # use override directly when provided; fall back to auto
            zoom_max = zoom_max_override if zoom_max_override is not None else auto_zoom
            full_max = data[y_col].max() * 1.12

            # Floor of upper panel: just below first outlier above zoom_max
            above = data[y_col][data[y_col] > zoom_max].dropna()
            y_hi_min = zoom_max

            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(16, 10), sharex=True,
                gridspec_kw={
                    'height_ratios': [0.25, 0.75],
                    'hspace': 0.06
                }
            )
            sns.set_style("white")
            _plot_on(ax_top)
            _plot_on(ax_bot)

            ax_bot.set_ylim(0, zoom_max)
            ax_top.set_ylim(y_hi_min, full_max)

            # Ticks to nearest tenth, prune edges near the break
            # choose decimal places based on the scale of the zoomed panel
            if zoom_max < 1:
                fmt = plt.FormatStrFormatter('%.2f')
            else:
                fmt = plt.FormatStrFormatter('%.1f')
            ax_bot.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune='upper'))
            ax_bot.yaxis.set_major_formatter(fmt)
            ax_top.yaxis.set_major_locator(plt.MaxNLocator(nbins=2, prune='lower'))
            ax_top.yaxis.set_major_formatter(fmt)

            # Spine styling — clean break, no decorations
            for ax in (ax_top, ax_bot):
                ax.spines['right'].set_visible(False)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['bottom'].set_visible(False)
            ax_top.spines['left'].set_color('black')
            ax_bot.spines['top'].set_visible(False)
            ax_bot.spines['bottom'].set_color('black')
            ax_bot.spines['left'].set_color('black')

            # Tick / label styling — rotate x labels to prevent overlap
            ax_top.tick_params(axis='x', length=0, labelbottom=False)
            ax_top.tick_params(axis='y', colors='black', length=7, width=1.5, labelsize=26)
            ax_bot.tick_params(axis='y', colors='black', length=7, width=1.5, labelsize=26)
            ax_bot.tick_params(axis='x', colors='black', length=7, width=1.5, labelsize=26)
            plt.setp(ax_bot.get_xticklabels(), rotation=45, ha='right')

            ax_bot.set_xlabel(xlabel, fontsize=30, color='black', fontweight='bold')
            ax_top.set_xlabel("")
            fig.text(0.02, 0.5, f"{level_name} {ylabel_base} (Hz)",
                     va='center', rotation='vertical',
                     fontsize=30, color='black', fontweight='bold')

            ax_top.text(0.97, 0.95,
                        f"{traj_summary}\n{posthoc_text}".strip(),
                        transform=ax_top.transAxes,
                        fontsize=20, fontweight='bold', ha='right', va='top',
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.set_style("white")
            _plot_on(ax)
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(f"{level_name} {ylabel_base} (Hz)",
                          fontsize=30, color='black', fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=30, color='black',
                          fontweight='bold')
            ax.tick_params(axis='both', colors='black', length=7,
                           width=1.5, labelsize=26)
            ax.text(0.97, 0.95,
                    f"{traj_summary}\n{posthoc_text}".strip(),
                    transform=ax.transAxes,
                    fontsize=20, fontweight='bold', ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.85,
                              edgecolor='gray'))

        if not split_axis:
            plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir,
                                 f"{feature}_{level_name.replace(' ', '_').lower()}.pdf")
            plt.savefig(fname, bbox_inches='tight', transparent=True)
        plt.show()

    draw_trajectory_panel(df, "plot_val", "Well-Level")

    if elec_key and elec_key in df.columns:
        expanded = []
        for _, row in df.iterrows():
            if isinstance(row[elec_key], (list, np.ndarray)):
                for v in row[elec_key]:
                    expanded.append({"group": row["group"], "time_val": row["time_val"], "age_days": row["age_days"], "v": v})
        
        if expanded:
            elec_df = pd.DataFrame(expanded).dropna()
            if feature == "firing": elec_df = elec_df[elec_df["v"] >= threshold_hz]
            zoom_overrides = {"firing": 11, "burst": 0.25}
            zoom_override = zoom_overrides.get(feature, None)
            draw_trajectory_panel(elec_df, "v", "Electrode-Level", broken_axis=True, zoom_max_override=zoom_override)


def format_table_with_stars(df):
    df_stars = df.copy()
    
    comp_cols = [c for c in df_stars.columns if " vs " in c]
    
    for col in comp_cols:
        df_stars[col] = df_stars[col].apply(get_stars)
    
    if "ANOVA_p" in df_stars.columns:
        df_stars["ANOVA_p"] = df_stars["ANOVA_p"].apply(get_stars)
        
    return df_stars


def plot_grouped_genotype_analysis(df, feature, palette, order, threshold_spikes_per_min=5, save_dir=None):
    df = df.copy()
    threshold_hz = threshold_spikes_per_min / 60.0
    FONT_SIZE = 20

    def draw_grouped_panel(data, y_col, level_name, feat_label):
        data = data.dropna(subset=[y_col])
        if data.empty: return
        
        try:
            comp = mc.MultiComparison(data[y_col], data["group"])
            ph_df = pd.DataFrame(data=comp.tukeyhsd().summary().data[1:], 
                                 columns=comp.tukeyhsd().summary().data[0])
        except Exception as e:
            print(f"Stats error in {level_name} {feat_label}: {e}")
            ph_df = pd.DataFrame()

        fig, ax = plt.subplots(figsize=(4, 6))
        sns.set_style("white")
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

        group_idx = {name: i for i, name in enumerate(order)}
        x_positions = np.arange(len(order))
        bar_width = 0.5

        # Draw mean bars + SEM error bars
        for i, g in enumerate(order):
            sub = data[data["group"] == g][y_col].dropna()
            mean_val = sub.mean()
            sem_val = sub.sem()
            ax.bar(i, mean_val, width=bar_width, color=palette.get(g, "#AAAAAA"),
                   alpha=0.9, edgecolor='black', linewidth=0.8, zorder=2)
            ax.errorbar(i, mean_val, yerr=sem_val, fmt='none', color='black',
                        capsize=4, capthick=1.2, elinewidth=1.2, zorder=4)

        # Individual data points overlaid
        rng = np.random.default_rng(42)
        for i, g in enumerate(order):
            sub = data[data["group"] == g][y_col].dropna()
            jitter = rng.uniform(-0.15, 0.15, size=len(sub))
            ax.scatter(i + jitter, sub.values, color='#555555', alpha=0.65, s=20,
                       edgecolors='none', zorder=5)

        ax.set_xticks(x_positions)
        # Color each x-tick label to match its bar color (like the reference figure)
        ax.set_xticklabels(order, fontsize=FONT_SIZE - 2, rotation=45, ha='right')
        for tick_label, g in zip(ax.get_xticklabels(), order):
            tick_label.set_color(palette.get(g, 'black'))
            tick_label.set_fontweight('bold')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Hide x-axis tick marks; keep only y-axis ticks
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', colors='black', length=5, width=1)

        # Significance brackets — sort by span (adjacent pairs first)
        all_vals = data[y_col].dropna()
        y_max_data = all_vals.max()
        y_min_data = all_vals.min()
        y_range = y_max_data - y_min_data if y_max_data != y_min_data else abs(y_max_data)

        bracket_level = y_max_data + y_range * 0.08
        step = y_range * 0.14

        if not ph_df.empty:
            pairs = []
            for _, row in ph_df.iterrows():
                g1, g2 = row['group1'], row['group2']
                p_adj = row['p-adj']
                if g1 in group_idx and g2 in group_idx:
                    span = abs(group_idx[g1] - group_idx[g2])
                    pairs.append((span, group_idx[g1], group_idx[g2], p_adj))
            pairs.sort()

            for _, x1, x2, p_adj in pairs:
                stars = get_stars(p_adj)
                tip_h = step * 0.12
                ax.plot([x1, x1, x2, x2],
                        [bracket_level, bracket_level + tip_h,
                         bracket_level + tip_h, bracket_level],
                        lw=1.2, color='black', zorder=6)
                ax.text((x1 + x2) * 0.5, bracket_level + tip_h * 1.1, stars,
                        ha='center', va='bottom', color='black',
                        fontsize=FONT_SIZE + 2, fontweight='bold')
                bracket_level += step

        ax.set_ylim(bottom=0, top=bracket_level + step * 0.5)

        if "Synchrony" in feat_label:
            unit_label = ""
        else:
            unit_label = "(Hz)"

        plt.ylabel(f"{feat_label}\n{unit_label}", fontsize=FONT_SIZE + 2,
                   color='black', fontweight='bold', labelpad=8)
        plt.yticks(fontsize=FONT_SIZE, color='black')

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"grouped_{feat_label.replace(' ', '_').lower()}_{level_name.replace(' ', '_').lower()}.pdf")
            plt.savefig(fname, bbox_inches='tight', transparent=True)
        plt.show()

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


def run_mea_analysis_for_figure(df, experiment, unit="days", palette=MAOA_FIGURE_PALETTE, threshold_spikes_per_min=5, save_dir=None):
    df_exp = df[df["experiment"] == experiment].copy()

    def assign_genotype(line):
        if line in {"C30"}: return "Xa:WT/Xi:MUT"
        if line in {"C71", "C3"}: return "Xa:MUT/Xi:WT"
        if line in {"GM", "MAOA/B−/Y"}: return "MAOA/B$^{-/Y}$"
        return "other"

    df_exp["group"] = df_exp["cell_line"].apply(assign_genotype)

    ref_date = pd.to_datetime("2025-08-12")
    df_exp["recording_date"] = pd.to_datetime(df_exp["recording_date"])
    df_exp["age_days"] = (df_exp["recording_date"] - ref_date).dt.days + 180
    df_exp["time_val"] = (df_exp["age_days"] / 30.44).round(1) if unit == "months" else df_exp["age_days"]

    for feat in ["firing", "burst", "network", "synchrony"]:
        print(f"\n{'='*40}\nFEATURE: {feat.upper()}\n{'='*40}")
        plot_figure_boxplots_for_paper(df_exp, feat, palette, PLOT_ORDER, unit, save_dir=save_dir)
        
        print("\n--- WELL-LEVEL STATS ---")
        well_stats = run_comprehensive_stats(df_exp, feat, level="well", unit=unit)
        print("ANOVA Interaction:\n", well_stats["anova"])
        
        print("\nTime-Point Table (Significance Stars):")
        print(format_table_with_stars(well_stats["table"]).to_string(index=False))
        
        if feat in ["firing", "burst"]:
            print(f"\n--- ELECTRODE-LEVEL {feat.upper()} STATS ---")
            elec_stats = run_comprehensive_stats(df_exp, feat, level="electrode", unit=unit)
            
            print("\nTime-Point Table (Significance Stars):")
            print(format_table_with_stars(elec_stats["table"]).to_string(index=False))

def run_grouped_analysis(df, experiment, save_dir=None):
    df_exp = df[df["experiment"] == experiment].copy()

    def assign_genotype(line):
        if line in {"C30", "C12c"}: return "Xa:WT/Xi:MUT"
        if line in {"C71", "C3"}: return "Xa:MUT/Xi:WT"
        if line in {"GM", "MAOA/B−/Y"}: return "MAOA/B$^{-/Y}$"
        return "other"

    df_exp["group"] = df_exp["cell_line"].apply(assign_genotype)
    df_exp = df_exp[df_exp["group"] != "other"]

    for feat in ["firing", "burst", "network", "synchrony"]:
        print(f"\n--- GENERATING GROUPED PLOTS FOR {feat.upper()} ---")
        plot_grouped_genotype_analysis(df_exp, feat, MAOA_FIGURE_PALETTE, PLOT_ORDER, save_dir=save_dir)

def find_best_single_window(well_spikes, goal="sync", duration=60):
    if well_spikes.empty: return 0
    t_max = well_spikes["time_s"].max()
    bins = np.arange(0, t_max, 0.05)
    counts, bin_edges = np.histogram(well_spikes["time_s"], bins=bins)
    
    window_bins = int(duration / 0.05)
    step_bins = int(5 / 0.05)
    
    best_start = 0
    best_score = -float('inf')
    
    for i in range(0, len(counts) - window_bins, step_bins):
        start_t = bin_edges[i]
        w_mask = (well_spikes["time_s"] >= start_t) & (well_spikes["time_s"] < start_t + duration)
        w_spikes = well_spikes[w_mask]
        
        total_spikes = len(w_spikes)
        active_elecs = w_spikes["electrode"].nunique()
        
        if active_elecs < 8 or total_spikes < 30: continue 
            
        w_counts = counts[i : i + window_bins]
        max_peak = np.max(w_counts)
        
        if goal == "sync":
            score = max_peak * active_elecs
        else:
            score = -max_peak 
            
        if score > best_score:
            best_score = score
            best_start = start_t
            
    return best_start if best_score != -float('inf') else well_spikes["time_s"].min()

def generate_activity_gallery(big_df, experiment="Exp3", num_options=5, duration=60):
    print(f"--- Generating Gallery (Low-Activity Mutants) for {experiment} ---")
    
    df_exp = big_df[big_df["experiment"] == experiment].copy()
    
    def assign_genotype(line):
        if line in {"C30", "C12c"}: return "Xa:WT/Xi:MUT"
        if line in {"C71", "C3"}: return "Xa:MUT/Xi:WT"
        if line in {"GM", "MAOA/B−/Y"}: return "MAOA/B$^{-/Y}$"
        return "other"

    df_exp["group"] = df_exp["cell_line"].apply(assign_genotype)
    df_exp["mean_fr"] = df_exp["firing_rate_stats_hz"].apply(lambda x: x[0] if isinstance(x, list) else np.nan)
    
    gallery_selections = {"Xa:WT/Xi:MUT": [], "Xa:MUT/Xi:WT": [], "MAOA/B$^{-/Y}$": []}
    
    ctrl_sub = df_exp[df_exp["group"] == "Xa:WT/Xi:MUT"].dropna(subset=["synchrony_index", "mean_fr"])
    if not ctrl_sub.empty:
        active_ctrl = ctrl_sub[ctrl_sub["mean_fr"] >= ctrl_sub["mean_fr"].quantile(0.50)]
        top_ctrls = active_ctrl.sort_values("synchrony_index", ascending=False).head(num_options)
        for _, row in top_ctrls.iterrows():
            gallery_selections["Xa:WT/Xi:MUT"].append({"row": row, "goal": "sync"})
        
    for mut in ["Xa:MUT/Xi:WT", "MAOA/B$^{-/Y}$"]:
        sub = df_exp[df_exp["group"] == mut].dropna(subset=["synchrony_index", "mean_fr"])
        if not sub.empty:
            lower_bound = sub["mean_fr"].quantile(0.15)
            upper_bound = sub["mean_fr"].quantile(0.50)
            
            target_muts = sub[(sub["mean_fr"] >= lower_bound) & (sub["mean_fr"] <= upper_bound)]
            
            if len(target_muts) < num_options:
                target_muts = sub[sub["mean_fr"] >= lower_bound]
                
            top_muts = target_muts.sort_values("synchrony_index", ascending=True).head(num_options)
            for _, row in top_muts.iterrows():
                gallery_selections[mut].append({"row": row, "goal": "async_low"})

    groups = ["Xa:WT/Xi:MUT", "Xa:MUT/Xi:WT", "MAOA/B$^{-/Y}$"]
    fig, axes = plt.subplots(3, num_options, figsize=(5 * num_options, 12), sharex=True, sharey=True)
    
    for row_idx, group in enumerate(groups):
        options = gallery_selections[group]
        color = MAOA_FIGURE_PALETTE[group]
        
        for col_idx in range(num_options):
            ax = axes[row_idx, col_idx]
            
            if col_idx >= len(options):
                ax.axis('off') 
                continue
                
            info = options[col_idx]
            row_data = info["row"]
            well = row_data["well"]
            plate = row_data["plate_name"]
            
            spikes = load_spike_list(row_data["spike_list_path"])
            well_spikes = spikes[spikes["well"] == well]
            t_tot = well_spikes["time_s"].max() - well_spikes["time_s"].min()
            if t_tot > 0:
                c = well_spikes["electrode"].value_counts()
                well_spikes = well_spikes[well_spikes["electrode"].isin(c[c / t_tot < 15].index)]
                
            start_t = find_best_single_window(well_spikes, goal=info["goal"], duration=duration)
            end_t = start_t + duration
            
            w_spikes = well_spikes[(well_spikes["time_s"] >= start_t) & (well_spikes["time_s"] < end_t)]
            expected = [f"{well}_{r}{c}" for r in range(1, 9) for c in range(1, 9)]
            trains = [w_spikes[w_spikes["electrode"] == e]["time_s"] - start_t for e in expected]
            
            ax.eventplot(trains, color='black', linelengths=0.9, linewidths=1.2)
            
            title_str = f"{group}\n{plate} | Well {well} | Start: {start_t:.0f}s"
            ax.set_title(title_str, color=color, fontweight='bold', fontsize=12)
            ax.set_ylim(-1, 64)
            ax.set_xlim(0, duration)
            
            if col_idx == 0:
                ax.set_ylabel("Electrode Unit", fontsize=14)
                ax.set_yticks([0, 15, 31, 47, 63])
                ax.set_yticklabels([1, 16, 32, 48, 64])
                
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (sec)", fontsize=14)
        
    sns.despine()
    plt.tight_layout()
    plt.show()