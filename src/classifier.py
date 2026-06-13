"""Coordinator for the Email Classifier.

This module connects together the main classifier components:
- Dataset loading -> CSV data ingestion
- Text vectorization -> TF-IDF feature extraction
- Model training -> Logistic Regression or Naive Bayes
- Prediction -> Classification with confidence scores
- Evaluation -> Accuracy and confusion matrix
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
 
from src.conversion import SpamDataset, TextVectorizer


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
        self.dataset = SpamDataset.load(knowledge_base_path)
        self.dataset["label"] = (
            self.dataset["label"].astype(str).str.strip().str.lower()
        )
        self.vectorizer = TextVectorizer() # Using custom class
        self.model: LogisticRegression | MultinomialNB | None = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained = False

    def get_dataset_info(self) -> str:
        """
        Get information about the loaded dataset.

        Returns:
            A printable dataset information summary.
        """
        label_dist = self.dataset["label"].value_counts().to_dict()
        avg_len = self.dataset["message"].str.len().mean()

        info = f"Total records: {len(self.dataset)}\n"
        info += f"Label distribution: {label_dist}\n"
        info += f"Average message length: {avg_len:.0f} characters"
        return info

    def train(self, model_type: str = "logistic", test_size: float = 0.2, random_state: int = 42) -> None:
        """Train the classifier using the loaded dataset."""
        if self.dataset.empty:
            raise ValueError("Cannot train classifier on an empty dataset.")

        # Use the TextVectorizer to fit and transform the data
        X = self.vectorizer.fit_transform(self.dataset["message"].astype(str))
        y = self.dataset["label"].astype(str)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        if model_type.lower() in {"logistic", "lr"}:
            self.model = LogisticRegression(max_iter=1000)
        elif model_type.lower() in {"naive_bayes", "nb", "multinomial_nb"}:
            self.model = MultinomialNB()
        else:
            raise ValueError("Unsupported model type. Choose 'logistic' or 'naive_bayes'.")

        self.model.fit(self.X_train, self.y_train)
        self.trained = True

    def predict(self, text: str) -> tuple[str, float]:
        """Predict the label and confidence for a single text input."""
        if not self.trained or self.model is None:
            raise ValueError("The classifier must be trained before making predictions.")

        text = str(text).strip()
        if not text:
            raise ValueError("Input message must not be empty.")

        matrix = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(matrix)[0]
        predicted_label = self.model.classes_[probabilities.argmax()]
        confidence = float(probabilities.max())
        return predicted_label, confidence

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the trained classifier on the holdout test set."""
        if not self.trained or self.model is None:
            raise ValueError("The classifier must be trained before evaluation.")

        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        confusion = confusion_matrix(self.y_test, predictions, labels=self.model.classes_)
        report = classification_report(self.y_test, predictions, zero_division=0)

        return {
            "accuracy": accuracy,
            "confusion_matrix": confusion.tolist(),
            "classes": list(self.model.classes_),
            "report": report,
        }
