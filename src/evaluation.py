"""Evaluation utilities for the email classifier.

This module provides a helper function to evaluate a trained classification
model on a holdout test set, producing accuracy and a confusion matrix
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test, spam_threshold: float = 0.50):
    """Evaluate a trained model using test data.

    Args:
        model: A trained scikit-learn classifier with predict_proba support.
        X_test: test messages converted into a matrix of numerical TF-IDF values
        y_test: correct labels for the test messages, like ham or spam
        spam_threshold: Probability threshold above which a message is labeled
            as spam.

    Returns:
        A dictionary containing accuracy, confusion matrix, classes
    """

    # converts label names into a normal Python list of strings
    classes = [str(label) for label in model.classes_]
    # finds the position of "spam" inside the class list
    spam_index = classes.index("spam")
    # asks the model to predict probabilities for every test message
    # Return a matrix of prob for each label ex: ["ham", "spam"] -> [0.95, 0.05]
    probabilities = model.predict_proba(X_test)

    # For each row of probabilities, checks the spam probability
    predictions = [
        # If the spam probability is greater than or equal to the threshold, predict "spam"
        "spam" if row[spam_index] >= spam_threshold else "ham"
        for row in probabilities
    ]

    # compares the real labels y_test with the predicted labels
    accuracy = accuracy_score(y_test, predictions)
    
    # creates a confusion matrix
    confusion = confusion_matrix(y_test, predictions, labels=classes)
    
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "classes": classes,
    }

def print_evaluation(evaluation: dict) -> None:
    """Print evaluation metrics for the trained classifier.

    Args:
        evaluation: A dictionary returned by EmailClassifier.evaluate(),
            containing accuracy, confusion matrix and labels
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
