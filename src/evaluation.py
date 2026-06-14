from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test, spam_threshold: float = 0.50):
    """Evaluate a trained model using test data."""

    classes = list(model.classes_)
    spam_index = classes.index("spam")
    probabilities = model.predict_proba(X_test)

    predictions = [
        "spam" if row[spam_index] >= spam_threshold else "ham"
        for row in probabilities
    ]

    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions, labels=classes)
    report = classification_report(y_test, predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "classes": classes,
        "report": report,
    }
