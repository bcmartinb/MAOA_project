import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Visualize well → cell line mapping for a plate as a 3×4 grid."""

    sub = df[df["plate_name"] == plate_name].copy()
    if sub.empty:
        print(f"No data found for plate {plate_name}")
        return
    
    # Keep ONE row per well (all dates should have same cell line)
    sub_unique = sub.drop_duplicates(subset=["well", "cell_line"]).copy()

    # Extract row letter and column number
    sub_unique["row"] = sub_unique["well"].str[0]
    sub_unique["col"] = sub_unique["well"].str[1:].astype(int)

    # Pivot to grid
    pivot = sub_unique.pivot(index="row", columns="col", values="cell_line")

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="s",
        cbar=False,
        linewidths=1,
        linecolor="black",
        cmap="Pastel1"
    )
    plt.title(f"{plate_name} — Well → Cell Line Map")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


