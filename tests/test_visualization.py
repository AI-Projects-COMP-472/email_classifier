import pandas as pd
import pytest

from src.visualization import plot_label_distribution


def test_plot_label_distribution_accepts_valid_dataset():
    """Test that a valid dataset can be plotted without displaying the chart."""

    dataset = pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "spam", "ham"],
            "message": [
                "hello friend",
                "win money now",
                "meeting today",
                "claim prize",
                "see you later",
            ],
        }
    )

    # show_plot=False keeps the test from opening a chart window.
    plot_label_distribution(dataset, show_plot=False)


def test_plot_label_distribution_saves_chart_file(tmp_path):
    """Test that the chart can be saved to a file."""

    dataset = pd.DataFrame(
        {
            "label": ["ham", "spam", "ham"],
            "message": [
                "hello friend",
                "win money now",
                "meeting today",
            ],
        }
    )
    output_file = tmp_path / "label_distribution.png"

    plot_label_distribution(dataset, save_path=output_file, show_plot=False)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_label_distribution_rejects_missing_label_column():
    """Test that a dataset without a label column raises a ValueError."""

    dataset = pd.DataFrame(
        {
            "message": [
                "hello friend",
                "win money now",
            ],
        }
    )

    with pytest.raises(ValueError):
        plot_label_distribution(dataset, show_plot=False)


def test_plot_label_distribution_rejects_empty_dataset():
    """Test that an empty dataset raises a ValueError."""

    dataset = pd.DataFrame(columns=["label", "message"])

    with pytest.raises(ValueError):
        plot_label_distribution(dataset, show_plot=False)