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

from src.dataset import SpamDataset
from src.vectorizer import TextVectorizer
from src.training import create_model, split_training_data, train_model
from src.evaluation import evaluate_model

class EmailClassifier:
    """
    Email spam classifier coordinator.

    This class separates the classifier logic from the command-line interface.
    """

    def __init__(
            self,
            dataset_path: str = "data/spam.csv",
            spam_threshold: float = 0.40
    ) -> None:
        """
        Initialize the email classifier.

        Args:
            dataset_path: Path to the CSV dataset file.
            spam_threshold: Minimum spam probability needed to classify a
                message as spam. Lower values catch spam more aggressively.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is invalid.
        """
        self.dataset = SpamDataset.load(dataset_path)
        
        self.dataset["label"] = (
            self.dataset["label"].astype(str).str.strip().str.lower()
        )

        if not 0.0 <= spam_threshold <= 1.0:
            raise ValueError("Spam threshold must be between 0.0 and 1.0.")

        self.spam_threshold = spam_threshold
        self.vectorizer = TextVectorizer() # Using custom class
        self.model: LogisticRegression | MultinomialNB | None = None
        
        # The training messages, converted into TF-IDF numerical features
        self.X_train = None # 80% of messages converted into numbers, used to train
        self.X_test = None # 20% of messages converted into numbers, used to test
        
        # The training labels, like ham or spam
        self.y_train = None # correct labels for those 80% messages
        self.y_test = None # correct labels for those 20% messages

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

        train_dataset, test_dataset, y_train, y_test = split_training_data(
            self.dataset,
            self.dataset["label"],
            test_size=test_size,
            random_state=random_state,
        )

        ''' So the model pipeline doesn't see the test data before evaluation '''
        # learns vocabulary only from the 80% training messages
        self.X_train = self.vectorizer.fit_transform(
            # Return matrix of TF-IDF values
            train_dataset["message"].astype(str)
        )
        
        # converts the 20% test messages using the training vocabulary, 
        # but does not learn from them
        self.X_test = self.vectorizer.transform(
            test_dataset["message"].astype(str)
        )

        # Assigning the splitted data "label" to coresponding train or test purpose
        self.y_train = y_train.astype(str)
        self.y_test = y_test.astype(str)

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
        
        classes = list(self.model.classes_)
        spam_index = classes.index("spam")
        spam_probability = float(probabilities[spam_index])

        # Debug
        # print(f"Spam probability: {spam_probability:.4f}")

        if spam_probability >= self.spam_threshold:
            predicted_label = "spam"
            confidence = spam_probability
        else:
            predicted_label = "ham"
            confidence = 1.0 - spam_probability
        
        return predicted_label, confidence

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the trained classifier on the holdout test set."""
        
        if not self.trained or self.model is None:
            raise ValueError("The classifier must be trained before evaluation.")

        return evaluate_model(
            self.model,
            self.X_test,
            self.y_test,
            spam_threshold=self.spam_threshold,
        )
