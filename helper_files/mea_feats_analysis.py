import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches


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



def plot_plate_map(df, plate_name):
    """Draw a 3×4 plate map showing cell line labels in each well."""

    sub = df[df["plate_name"] == plate_name].copy()
    if sub.empty:
        print(f"No data found for plate {plate_name}")
        return

    # Take one row per well (ignoring dates)
    sub = sub.drop_duplicates(subset=["well"]).copy()

    # Parse row and column
    sub["row"] = sub["well"].str[0]
    sub["col"] = sub["well"].str[1:].astype(int)

    # Order rows like A,B,C
    row_order = ["A", "B", "C"]
    sub["row_index"] = sub["row"].map({r:i for i,r in enumerate(row_order)})

    fig, ax = plt.subplots(figsize=(6,4))

    # categorical color map — stable colors per cell line
    cell_lines = sorted(sub["cell_line"].unique())
    color_map = {cl: plt.cm.Set3(i / len(cell_lines)) for i, cl in enumerate(cell_lines)}

    # Draw wells
    for _, r in sub.iterrows():
        x = r["col"] - 1       # 0-based
        y = r["row_index"]     # 0-based

        rect = patches.Rectangle(
            (x, y), 1, 1,
            linewidth=1,
            edgecolor='black',
            facecolor=color_map[r["cell_line"]]
        )
        ax.add_patch(rect)

        # Label inside the well
        ax.text(
            x + 0.5, y + 0.5,
            r["cell_line"],
            ha='center', va='center', fontsize=12
        )

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)

    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels([1,2,3,4])

    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(row_order)

    ax.set_title(f"{plate_name} — Cell Line Layout")
    ax.invert_yaxis()  # so A is at top
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_feature_boxplots(df, feature, threshold_spikes_per_min=5):
    """
    Plot boxplots for a given feature across experiments.

    feature options:
        "firing"      -> wells + electrodes (threshold applies)
        "burst"       -> wells + electrodes (NO threshold)
        "network"     -> wells only
        "synchrony"   -> wells only
    """

    # convert threshold to Hz
    threshold_hz = threshold_spikes_per_min / 60.0

    # Validate feature
    feature = feature.lower()
    valid = ["firing", "burst", "network", "synchrony"]
    if feature not in valid:
        raise ValueError(f"feature must be one of {valid}")

    # Extract well-level means
    if feature == "firing":
        df["well_value"] = df["firing_rate_stats_hz"].str[0]
    elif feature == "burst":
        df["well_value"] = df["burst_rate_stats_hz"].str[0]
    elif feature == "network":
        df["well_value"] = df["network_event_rate_hz"]
    elif feature == "synchrony":
        df["well_value"] = df["synchrony_index"]

    df["date_label"] = df["recording_date"].astype(str)

    # Loop per experiment
    for exp in sorted(df["experiment"].unique()):
        sub = df[df["experiment"] == exp].copy()
        print(f"\n=== Plotting Experiment {exp} ===")

        # ------------------ WELL-LEVEL ------------------
        plt.figure(figsize=(18,6))
        sns.boxplot(
            data=sub,
            x="date_label",
            y="well_value",
            hue="cell_line",
            showfliers=False,
            width=0.7
        )
        plt.title(f"Experiment {exp} — Well-Level {feature.capitalize()} Rate")
        plt.xlabel("Recording date")
        plt.ylabel(f"{feature.capitalize()} rate (Hz)")
        plt.xticks(rotation=45)
        plt.legend(title="Cell line", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        # ------------------ ELECTRODE-LEVEL ------------------
        if feature in ["firing", "burst"]:

            # Select correct vector column
            key = "firing_rate_vector_hz" if feature == "firing" else "burst_rate_vector_hz"

            # Build electrode-level dataframe
            def expand_electrodes(row):
                return pd.DataFrame({
                    "plate_name": row["plate_name"],
                    "well": row["well"],
                    "cell_line": row["cell_line"],
                    "recording_date": row["recording_date"],
                    "age_days": row["organoid_age_days"],
                    "electrode": row["electrodes"],
                    "value": row[key],
                })

            elec_df = pd.concat(
                [expand_electrodes(r) for _, r in sub.iterrows()],
                ignore_index=True
            )

            elec_df = elec_df.dropna(subset=["value"])
            elec_df["date_label"] = elec_df["recording_date"].astype(str)

            # Only apply threshold for FIRING
            if feature == "firing":
                elec_df = elec_df[elec_df["value"] >= threshold_hz]

            if elec_df.empty:
                print(f"No valid electrodes in Experiment {exp} for {feature}.")
                continue

            # Electrode-level boxplot
            plt.figure(figsize=(18,6))
            sns.boxplot(
                data=elec_df,
                x="date_label",
                y="value",
                hue="cell_line",
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
            plt.legend(title="Cell line", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.show()