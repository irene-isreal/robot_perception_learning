from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    input_path = Path("results/fuzzy_planner_comparison.csv")
    output_path = Path("results/fuzzy_planner_comparison.png")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = [
        "algorithm",
        "coverage",
        "frontiers_reached",
        "replans",
        "override_percentage",
    ]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df["algorithm"] = df["algorithm"].str.upper()

    expected_algorithms = {"BFS", "ASTAR"}
    if set(df["algorithm"].unique()) != expected_algorithms:
        raise ValueError("Expected exactly BFS and ASTAR in the evaluation CSV.")

    summary = df.groupby("algorithm")[required_columns[1:]].agg(["mean", "std"])

    metrics = [
        ("coverage", "Mean exploration coverage", "Coverage (%)"),
        ("frontiers_reached", "Frontiers reached", "Mean frontiers reached"),
        ("replans", "Replanning frequency", "Mean replans"),
        ("override_percentage", "Fuzzy safety overrides", "Overrides (%)"),
    ]

    algorithms = ["BFS", "ASTAR"]
    labels = ["BFS", "A*"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (column, title, ylabel) in zip(axes, metrics):
        means = [summary.loc[a, (column, "mean")] for a in algorithms]
        stds = [summary.loc[a, (column, "std")] for a in algorithms]

        ax.bar(labels, means, yerr=stds, capsize=5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        "BFS vs A* with Fuzzy Safety Control\n"
        "Mean and standard deviation across 30 episodes per algorithm",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison graph: {output_path}")
    print("\nEvaluation summary (mean ± standard deviation):")
    print(summary.round(3))


if __name__ == "__main__":
    main()
