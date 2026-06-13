"""Visualization helpers for the email classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def plot_label_distribution(
    dataset: pd.DataFrame,
    save_path: str | Path | None = None,
    show_plot: bool = True,
) -> None:
    """Plot the number of messages for each label in the dataset.

    Args:
        dataset: DataFrame containing at least a "label" column.
        save_path: Optional path where the chart should be saved.
        show_plot: Whether to display the chart window.

    Raises:
        ValueError: If the dataset is empty or missing the "label" column.
    """

    if "label" not in dataset.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    if dataset.empty:
        raise ValueError("Cannot plot label distribution for an empty dataset.")

    # Normalize labels before counting so values like "Spam" and " spam " match.
    label_counts = dataset["label"].astype(str).str.strip().str.lower().value_counts()

    figure, axis = plt.subplots(figsize=(6, 4))
    label_counts.plot(kind="bar", ax=axis, color=["#4C78A8", "#F58518"])

    axis.set_title("Message Label Distribution")
    axis.set_xlabel("Label")
    axis.set_ylabel("Number of messages")
    axis.tick_params(axis="x", rotation=0)

    # Add count labels above each bar for easier reading.
    for index, count in enumerate(label_counts):
        axis.text(index, count, str(count), ha="center", va="bottom")

    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close(figure)
