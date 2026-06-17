"""Training utilities for the email classifier.

This module provides helpers for creating the selected classifier model,
partitioning data into training and test sets, and fitting the model.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def create_model(model_type: str):
    """Create a supported machine learning model.

    Args:
        model_type: Name of the model to create, either 'logistic' or
            'naive_bayes'.

    Returns:
        An untrained scikit-learn classifier instance.

    Raises:
        ValueError: If the requested model type is not supported.
    """

    if model_type.lower() in {"logistic", "lr"}:
        return LogisticRegression(max_iter=1000)

    if model_type.lower() in {"naive_bayes", "nb", "multinomial_nb"}:
        return MultinomialNB()

    raise ValueError("Unsupported model type. Choose 'logistic' or 'naive_bayes'.")


def split_training_data(features, labels, test_size=0.2, random_state=42):
    """Split feature and label data into training and testing sets.

    Args:
        features: Feature matrix to split.
        labels: Target label array or Series.
        test_size: Fraction of data reserved for testing.
        random_state: RNG seed for reproducible splits.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """

    return train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )


def train_model(model, X_train, y_train):
    """Train a model using the provided training data.

    Args:
        model: A scikit-learn estimator with a fit method.
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        The trained model instance.
    """

    model.fit(X_train, y_train)
    return model