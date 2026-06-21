"""Training utilities for the email classifier.

This module provides helpers for creating the selected classifier model,
partitioning data into training and test sets, and fitting the model.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def create_model(model_type: str):
    """Create a machine learning classifier of the specified type.

    Creates an untrained model instance that can be fitted to training data.

    Args:
        model_type: Name of the model to create. Supported values:
                    - "logistic" or "lr" for Logistic Regression
                    - "naive_bayes" or "nb" for Multinomial Naive Bayes

    Returns:
        An untrained scikit-learn classifier instance.

    Raises:
        ValueError: If model_type is not supported.

    Example:
        >>> model = create_model("logistic")
        >>> model = create_model("naive_bayes")
    """

    if model_type.lower() in {"logistic", "lr"}:
        return LogisticRegression(max_iter=1000)

    if model_type.lower() in {"naive_bayes", "nb", "multinomial_nb"}:
        return MultinomialNB()

    raise ValueError("Unsupported model type. Choose 'logistic' or 'naive_bayes'.")


def split_training_data(features, labels, test_size=0.2, random_state=42):
    """Split feature and label data into training and testing sets.

    This function uses stratified splitting to ensure that both the training and test
    sets have roughly the same proportion of spam and ham messages as the original dataset.
    This is important for unbalanced datasets.

    Args:
        features: Feature matrix to split (e.g., TF-IDF vectors).
        labels: Target label array or Series (e.g., 'ham' or 'spam').
        test_size: Fraction of data to use for testing (default 0.2 = 20% test, 80% train).
        random_state: Random seed for reproducibility (default 42).
    
    Returns:
        A tuple of (features_train, features_test, labels_train, labels_test).
    """

    return train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,  # Keeps the same spam/ham ratio in both sets
        random_state=random_state,  # Same split every time for reproducibility
        shuffle=True,  # Mix rows before splitting (in case CSV is ordered by label)
    )


def train_model(model, X_train, y_train):
    """Train a scikit-learn classifier on the provided training data.

    Fits the model to learn patterns from the training features and labels.

    Args:
        model: A scikit-learn estimator with a fit() method
               (e.g., LogisticRegression, MultinomialNB).
        X_train: Training feature matrix (typically TF-IDF vectors).
        y_train: Training labels (array of class names).

    Returns:
        The trained model instance.
    """

    model.fit(X_train, y_train)
    return model