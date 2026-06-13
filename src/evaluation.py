from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model using test data."""

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions, labels=model.classes_)
    report = classification_report(y_test, predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist(),
        "classes": list(model.classes_),
        "report": report,
    }