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
    """Email spam classifier coordinator.

    This class connects all the machine learning pipeline components:
    - Dataset loading and management
    - Text vectorization (TF-IDF feature extraction)
    - Model training and selection
    - Prediction with confidence scores
    - Performance evaluation

    Example:
        >>> classifier = EmailClassifier(dataset_path="data/spam.csv")
        >>> classifier.train(model_type="logistic")
        >>> label, confidence = classifier.predict("Free money now!")
        >>> print(f"{label}: {confidence:.2%}")
        spam: 92.34%
    """
    # Constructor
    def __init__(
            self,
            dataset_path: str = "data/spam.csv",
            spam_threshold: float = 0.40
    ) -> None:
        """Initialize the email classifier.

        Loads a CSV dataset, initializes a vectorizer and prepares for model training.

        Args:
            dataset_path: Path to the CSV dataset file (default: "data/spam.csv").
                         Must have columns: label, message
            spam_threshold: Probability threshold for classifying as spam (default: 0.40).
                           Range: 0.0 to 1.0. Lower values catch more spam but may increase false positives.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is invalid or spam_threshold is out of range.
        """
        self.dataset = SpamDataset.load(dataset_path)
        
        # Normalize labels: convert to string, strip whitespace, convert to lowercase
        self.dataset["label"] = (
            self.dataset["label"].astype(str).str.strip().str.lower()
        )

        # Validate spam_threshold is in valid probability range
        if not 0.0 <= spam_threshold <= 1.0:
            raise ValueError("Spam threshold must be between 0.0 and 1.0.")

        self.spam_threshold = spam_threshold
        self.vectorizer = TextVectorizer()  # TF-IDF vectorizer
        self.model: LogisticRegression | MultinomialNB | None = None
        
        # Training/testing data splits - populated during train()
        self.X_train = None  # TF-IDF feature matrix for training set (80%)
        self.X_test = None   # TF-IDF feature matrix for test set (20%)
        self.y_train = None  # Labels for training set
        self.y_test = None   # Labels for test set

        self.trained = False

    def get_dataset_info(self) -> str:
        """Return a human-readable summary of the loaded dataset.

        Returns:
            A formatted string containing total records, label distribution,
            and average message length.
        """

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
        """Train the classifier on the loaded dataset.

        Splits the dataset into training (80%) and test (20%) sets, learns the
        TF-IDF vocabulary from training messages, and trains the selected model.

        Args:
            model_type: Type of model to train ("logistic" or "naive_bayes").
                       Default: "logistic"
            test_size: Fraction of data to use for testing (default: 0.2 = 20%).
            random_state: Random seed for reproducibility (default: 42).

        Raises:
            ValueError: If dataset is empty or model_type is unsupported.
        """
        
        if self.dataset.empty:
            raise ValueError("Cannot train classifier on an empty dataset.")

        # Split dataset into training and test sets
        train_dataset, test_dataset, y_train, y_test = split_training_data(
            self.dataset,
            self.dataset["label"],
            test_size=test_size,
            random_state=random_state,
        )

        # Learn TF-IDF vocabulary from training messages only (avoid data leakage)
        self.X_train = self.vectorizer.fit_transform(
            train_dataset["message"].astype(str)
        )
        
        # Transform test messages using the training vocabulary (no learning)
        self.X_test = self.vectorizer.transform(
            test_dataset["message"].astype(str)
        )

        # Store labels as strings
        self.y_train = y_train.astype(str)
        self.y_test = y_test.astype(str)

        # Create and train the selected model
        self.model = create_model(model_type)
        train_model(self.model, self.X_train, self.y_train)
        
        self.trained = True

    def predict(self, input: str) -> tuple[str, float]:
        """Predict the label and confidence score for one email message.

        Args:
            input: Email message text to classify.

        Returns:
            A tuple of (predicted_label, confidence) where:
            - predicted_label: "spam" or "ham"
            - confidence: float between 0.0 and 1.0 (higher = more certain)

        Raises:
            ValueError: If classifier is not trained or input is empty.

        Example:
            >>> classifier.train(model_type="logistic")
            >>> label, confidence = classifier.predict("Free money!")
            >>> print(f"{label}: {confidence:.2%}")
            spam: 92.34%
        """
        # Verify classifier has been trained
        if not self.trained or self.model is None:
            raise ValueError("The classifier must be trained before making predictions.")

        # Clean input: convert to string and strip whitespace
        input = str(input).strip()
        
        # Reject empty messages
        if not input:
            raise ValueError("Input message must not be empty.")

        # Convert text to TF-IDF features using learned vocabulary
        features = self.vectorizer.transform([input])
        # Get probability predictions for each class
        probabilities = self.model.predict_proba(features)[0]
        
        # Get class labels from the trained model
        classes = list(self.model.classes_)
        # Find the index of "spam" class
        spam_index = classes.index("spam")
        # Extract spam probability
        spam_probability = float(probabilities[spam_index])

        # Classify based on threshold
        if spam_probability >= self.spam_threshold:
            predicted_label = "spam"
            confidence = spam_probability
        else:
            predicted_label = "ham"
            # Confidence for ham is 1 - spam_probability
            confidence = 1.0 - spam_probability
        
        return predicted_label, confidence

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the trained classifier on the holdout test set.

        Computes accuracy and confusion matrix on test data using the trained model
        and the configured spam_threshold.

        Returns:
            A dictionary with keys:
                - 'accuracy': float between 0.0 and 1.0
                - 'confusion_matrix': list[list[int]]
                - 'classes': list[str] (class names)

        Raises:
            ValueError: If classifier is not trained.
        """
        
        if not self.trained or self.model is None:
            raise ValueError("The classifier must be trained before evaluation.")

        return evaluate_model(
            self.model,
            self.X_test,
            self.y_test,
            spam_threshold=self.spam_threshold,
        )
