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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.conversion import SpamDataset, TextVectorizer
from src.training import create_model, split_training_data, train_model
from src.evaluation import evaluate_model

class EmailClassifier:
    """
    Email spam classifier coordinator.

    This class separates the classifier logic from the command-line interface.
    """

    def __init__(self, dataset_path: str = "data/spam.csv") -> None:
        """
        Initialize the email classifier.

        Args:
            dataset_path: Path to the CSV dataset file.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is invalid.
        """
        self.dataset = SpamDataset.load(dataset_path)
        
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
        """Return a printable summary of the loaded dataset."""

        label_dist = self.dataset["label"].value_counts().to_dict()
        avg_len = self.dataset["message"].str.len().mean()

        info = f"Total records: {len(self.dataset)}\n"
        info += f"Label distribution: {label_dist}\n"
        info += f"Average message length: {avg_len:.0f} characters"
        
        return info

    def train(
            self, 
            model_type: str = "logistic", 
            test_size: float = 0.2, 
            random_state: int = 42
    ) -> None:
        """Train the classifier using the loaded dataset."""
        
        if self.dataset.empty:
            raise ValueError("Cannot train classifier on an empty dataset.")

        # Convert raw email text into TF-IDF numerical features using the TextVectorizer
        features = self.vectorizer.fit_transform(self.dataset["message"].astype(str))
        labels = self.dataset["label"].astype(str)

        self.X_train, self.X_test, self.y_train, self.y_test = split_training_data(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
        )

        self.model = create_model(model_type)
        train_model(self.model, self.X_train, self.y_train)
        
        self.trained = True

    def predict(self, text: str) -> tuple[str, float]:
        """Predict the label and confidence for score for one email message."""
        
        if not self.trained or self.model is None:
            raise ValueError("The classifier must be trained before making predictions.")

        text = str(text).strip()
        
        if not text:
            raise ValueError("Input message must not be empty.")

        features = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(features)[0]
        
        predicted_label = self.model.classes_[probabilities.argmax()]
        confidence = float(probabilities.max())
        
        return predicted_label, confidence

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the trained classifier on the holdout test set."""
        
        if not self.trained or self.model is None:
            raise ValueError("The classifier must be trained before evaluation.")

        return evaluate_model(self.model, self.X_test, self.y_test)
