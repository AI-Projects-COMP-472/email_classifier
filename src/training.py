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
        test_size: keeps 20% of the rows for testing, so the remaining 80% becomes self.train_dataset
        random_state: makes the split repeatable. Every time the program runs, it gets the same 80/20 split

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """

    return train_test_split(
        features,
        labels,
        test_size=test_size,
        
        # keeps roughly the same spam/ham ratio in both datasets. 
        # So if the full dataset is mostly ham, the training and testing sets will also be mostly ham.
        stratify=labels,
        
        random_state=random_state,
        
        # mixes the rows before splitting. In case the CSV is ordered in some pattern.
        shuffle=True,
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