"""Evaluation utilities for the email classifier.

This module provides a helper function to evaluate a trained classification
model on a holdout test set, producing accuracy and a confusion matrix
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test, spam_threshold: float = 0.50):
    """Evaluate a trained classifier model on unseen test data.

    This function compares the model's predictions against the true labels
    and computes evaluation metrics like accuracy and confusion matrix.
    The spam_threshold parameter controls the decision boundary: messages with
    spam probability >= threshold are classified as spam, otherwise as ham.

    Args:
        model: A trained scikit-learn classifier with predict_proba() support
               (e.g., LogisticRegression, MultinomialNB).
        X_test: Test feature matrix (TF-IDF vectors from vectorizer.transform()).
        y_test: True labels for test set (array of 'ham' or 'spam' strings).
        spam_threshold: Probability threshold for classifying as spam.
                       Default is 0.50 (neutral). Lower values are more aggressive.

    Returns:
        A dictionary with keys:
            - 'accuracy': float between 0 and 1
            - 'confusion_matrix': list[list[int]] (rows=true labels, cols=predicted)
            - 'classes': list[str] (the class names, e.g., ['ham', 'spam'])
    """

    # Convert class names to strings
    classes = [str(label) for label in model.classes_]
    # Find the index of the "spam" class
    spam_index = classes.index("spam")
    # Get probability predictions for each test message
    probabilities = model.predict_proba(X_test)

    # Make predictions using the threshold
    predictions = [
        "spam" if row[spam_index] >= spam_threshold else "ham"
        for row in probabilities
    ]

    # Calculate accuracy by comparing predictions to true labels
    accuracy = accuracy_score(y_test, predictions)
    
    # Create confusion matrix showing true vs predicted labels
    confusion = confusion_matrix(y_test, predictions, labels=classes)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "classes": classes,
    }

def print_evaluation(evaluation: dict) -> None:
    """Print evaluation metrics in a human-readable format.

    Displays accuracy percentage and a formatted confusion matrix showing
    how many messages were classified correctly vs incorrectly.

    Args:
        evaluation: Dictionary returned by evaluate_model() containing:
                    - 'accuracy': accuracy score
                    - 'confusion_matrix': confusion matrix
                    - 'classes': class names

    Example:
        >>> evaluation = classifier.evaluate()
        >>> print_evaluation(evaluation)
        --- Evaluation ---
        Accuracy: 95.32%
        Confusion matrix:
          Classes: ['ham', 'spam']
          ham: [485, 12]
          spam: [8, 356]
    """
    classes = evaluation["classes"]
    confusion = evaluation["confusion_matrix"]
    print("\n--- Evaluation ---")
    print(f"Accuracy: {evaluation['accuracy']:.2%}")
    print("Confusion matrix:")
    print(f"  Classes: {classes}")
    for label, row in zip(classes, confusion):
        print(f"  {label}: {row}")
    print(f"{'---' * 8}\n")
