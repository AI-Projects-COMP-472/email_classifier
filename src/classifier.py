"""Coordinator for the Email Classifier.

This module connects together the main classifier components:
- Dataset loading -> CSV data ingestion
- Text vectorization -> TF-IDF feature extraction
- Model training -> Logistic Regression or Naive Bayes
- Prediction -> Classification with confidence scores
- Evaluation -> Accuracy and confusion matrix
- Conversation history + basic session stats
"""

from __future__ import annotations

from typing import List

import pandas as pd

from conversion import SpamDataset


class EmailClassifier:
    """
    Email spam classifier coordinator.

    This class separates the classifier logic from the command-line interface.
    """

    def __init__(self, knowledge_base_path: str = "data/spam.csv") -> None:
        """
        Initialize the email classifier.

        Args:
            knowledge_base_path: Path to the CSV dataset file.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is invalid.
        """
        # Load the dataset -> expects label and message columns
        self.dataset = SpamDataset.load(knowledge_base_path)

    def get_dataset_info(self) -> str:
        """
        Get information about the loaded dataset.

        Returns:
            A printable dataset information summary.
        """
        label_dist = self.dataset['label'].value_counts().to_dict()
        avg_len = self.dataset['message'].str.len().mean()

        info = f"Total records: {len(self.dataset)}\n"
        info += f"Label distribution: {label_dist}\n"
        info += f"Average message length: {avg_len:.0f} characters"
        return info

    # TODO: Add TextVectorizer for feature extraction
    # TODO: Add model training logic
    # TODO: Add prediction logic
    # TODO: Add evaluation logic