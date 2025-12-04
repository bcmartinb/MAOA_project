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


def plot_plate_map(df, plate_name, use_group3=False, double_label=False):
    """
    Draw a 3×4 plate map.

    Modes:
      - Default:     cell_line only  (original behavior)
      - use_group3:  shows group3 ONLY (Mutant / WT / GM)
      - double_label: shows both group3 and cell_line

    Priority:
      double_label=True overrides use_group3.
    """

    sub = df[df["plate_name"] == plate_name].copy()
    if sub.empty:
        print(f"No data found for plate {plate_name}")
        return

    # one row per well
    sub = sub.drop_duplicates(subset=["well"]).copy()

    # parse row/col
    sub["row"] = sub["well"].str[0]
    sub["col"] = sub["well"].str[1:].astype(int)

    row_order = ["A", "B", "C"]
    sub["row_index"] = sub["row"].map({r: i for i, r in enumerate(row_order)})

    fig, ax = plt.subplots(figsize=(6, 4))

    # ------------------------------
    # COLOR MAP LOGIC
    # ------------------------------
    if double_label or use_group3:
        # group colors
        color_map = {
            "Mutant": "#FF9999",
            "WT":     "#99CCFF",
            "GM":     "#CCFFCC"
        }
        group_values = sub["group3"].unique()
        for g in group_values:
            if g not in color_map:
                color_map[g] = "#DDDDDD"  # fallback
    else:
        # original cell line colors
        cell_lines = sorted(sub["cell_line"].unique())
        color_map = {cl: plt.cm.Set3(i / len(cell_lines))
                     for i, cl in enumerate(cell_lines)}

    # ------------------------------
    # DRAW WELLS
    # ------------------------------
    for _, r in sub.iterrows():
        x = r["col"] - 1
        y = r["row_index"]

        # choose fill color
        fill_key = r["group3"] if (use_group3 or double_label) else r["cell_line"]
        rect = patches.Rectangle(
            (x, y), 1, 1,
            linewidth=1,
            edgecolor="black",
            facecolor=color_map[fill_key]
        )
        ax.add_patch(rect)

        # TEXT LABELS
        if double_label:
            # Top: group3
            ax.text(
                x + 0.5, y + 0.40,
                r["group3"],
                ha="center", va="center",
                fontsize=12, fontweight="bold"
            )
            # Bottom: cell line
            ax.text(
                x + 0.5, y + 0.70,
                f"({r['cell_line']})",
                ha="center", va="center",
                fontsize=10
            )

        elif use_group3:
            ax.text(
                x + 0.5, y + 0.5,
                r["group3"],
                ha="center", va="center",
                fontsize=12, fontweight="bold"
            )

        else:
            ax.text(
                x + 0.5, y + 0.5,
                r["cell_line"],
                ha="center", va="center",
                fontsize=12
            )

    # Formatting
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels([1, 2, 3, 4])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(row_order)

    # Title
    if double_label:
        title = "Group + Cell Line Layout"
    elif use_group3:
        title = "Group Layout"
    else:
        title = "Cell Line Layout"

    ax.set_title(f"{plate_name} — {title}")
    ax.invert_yaxis()
    ax.set_aspect("equal")
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
    Runs TWO-WAY ANOVA (group_col × date_label) and ONE-WAY ANOVA,
    plus Bonferroni-corrected posthocs.
    Works for well-level and electrode-level (firing + burst).
    """

    df = df.copy()
    feature = feature.lower()
    threshold_hz = threshold_spikes_per_min / 60.0

    valid = ["firing", "burst", "network", "synchrony"]
    if feature not in valid:
        raise ValueError(f"feature must be: {valid}")

    # WELL LEVEL VALUES
    if feature == "firing":
        df["well_value"] = df["firing_rate_stats_hz"].str[0]
    elif feature == "burst":
        df["well_value"] = df["burst_rate_stats_hz"].str[0]
    elif feature == "network":
        df["well_value"] = df["network_event_rate_hz"]
    elif feature == "synchrony":
        df["well_value"] = df["synchrony_index"]

    df["date_label"] = df["recording_date"].astype(str)

    results = {}

    for exp in sorted(df["experiment"].unique()):
        sub = df[df["experiment"] == exp].dropna(subset=["well_value", group_col]).copy()
        exp_result = {"well": {}, "electrode": {}}

        # -------- WELL-LEVEL TWO-WAY ANOVA ----------
        if len(sub[group_col].unique()) > 1 and len(sub["date_label"].unique()) > 1:
            model = smf.ols(f"well_value ~ C({group_col}) * C(date_label)", data=sub).fit()
            anova_tw = anova_lm(model, typ=2)
            exp_result["well"]["two_way_anova"] = _clean_anova_table(anova_tw)
        else:
            exp_result["well"]["two_way_anova"] = None

        # -------- WELL-LEVEL ONE-WAY ANOVA ----------
        if len(sub[group_col].unique()) > 1:
            model = smf.ols(f"well_value ~ C({group_col})", data=sub).fit()
            anova_ow = anova_lm(model, typ=2)
            exp_result["well"]["one_way_anova"] = _clean_anova_table(anova_ow)
        else:
            exp_result["well"]["one_way_anova"] = None

        # -------- WELL-LEVEL POSTHOC ----------
        posthoc_list = []
        groups = sorted(sub[group_col].unique())
        from itertools import combinations
        pairs = list(combinations(groups, 2))
        m = len(pairs)

        for a, b in pairs:
            xa = sub[sub[group_col] == a]["well_value"].dropna()
            xb = sub[sub[group_col] == b]["well_value"].dropna()
            if len(xa) < 2 or len(xb) < 2:
                continue
            t, p = ttest_ind(xa, xb, equal_var=False)
            p_adj = min(p * m, 1.0)
            posthoc_list.append({
                "group1": a,
                "group2": b,
                "mean1": float(xa.mean()),
                "mean2": float(xb.mean()),
                "difference": float(xa.mean() - xb.mean()),
                "t_stat": float(t),
                "p_raw": float(p),
                "p_raw_fmt": _format_p(p),
                "p_bonf": float(p_adj),
                "p_bonf_fmt": _format_p(p_adj)
            })

        exp_result["well"]["posthoc"] = pd.DataFrame(posthoc_list) if posthoc_list else None

        # ======================================================
        # ELECTRODE LEVEL — only firing + burst
        # ======================================================
        if feature in ["firing", "burst"]:

            key = "firing_rate_vector_hz" if feature == "firing" else "burst_rate_vector_hz"

            def expand(row):
                return pd.DataFrame({
                    group_col: row[group_col],
                    "date_label": str(row["recording_date"]),
                    "value": row[key]
                })

            elec_df = pd.concat([expand(r) for _, r in sub.iterrows()], ignore_index=True)
            elec_df = elec_df.dropna(subset=["value"])

            if feature == "firing":
                elec_df = elec_df[elec_df["value"] >= threshold_hz]

            if elec_df.empty or len(elec_df[group_col].unique()) < 2:
                exp_result["electrode"] = {
                    "two_way_anova": None,
                    "one_way_anova": None,
                    "posthoc": None
                }
            else:
                # 2-way ANOVA
                if len(elec_df["date_label"].unique()) > 1:
                    model = smf.ols(f"value ~ C({group_col}) * C(date_label)", data=elec_df).fit()
                    anova_tw = anova_lm(model, typ=2)
                    exp_result["electrode"]["two_way_anova"] = _clean_anova_table(anova_tw)
                else:
                    exp_result["electrode"]["two_way_anova"] = None

                # 1-way ANOVA
                model = smf.ols(f"value ~ C({group_col})", data=elec_df).fit()
                anova_ow = anova_lm(model, typ=2)
                exp_result["electrode"]["one_way_anova"] = _clean_anova_table(anova_ow)

                # posthoc
                post_list = []
                groups_e = sorted(elec_df[group_col].unique())
                pairs_e = list(combinations(groups_e, 2))
                m_e = len(pairs_e)

                for a, b in pairs_e:
                    xa = elec_df[elec_df[group_col] == a]["value"]
                    xb = elec_df[elec_df[group_col] == b]["value"]
                    if len(xa) < 2 or len(xb) < 2:
                        continue
                    t, p = ttest_ind(xa, xb, equal_var=False)
                    p_adj = min(p * m_e, 1.0)
                    post_list.append({
                        "group1": a,
                        "group2": b,
                        "mean1": float(xa.mean()),
                        "mean2": float(xb.mean()),
                        "difference": float(xa.mean() - xb.mean()),
                        "t_stat": float(t),
                        "p_raw": float(p),
                        "p_raw_fmt": _format_p(p),
                        "p_bonf": float(p_adj),
                        "p_bonf_fmt": _format_p(p_adj)
                    })

                exp_result["electrode"]["posthoc"] = pd.DataFrame(post_list)

        results[exp] = exp_result

    return results


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



def assign_group3(line):
    mutant_lines = {"C3", "C71", "C90"}
    wt_lines = {"C12", "C30", "C54"}
    if line in mutant_lines:
        return "Mutant"
    elif line in wt_lines:
        return "WT"
    else:
        return "GM"



