from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluation import evaluate_model
from src.training import create_model, split_training_data, train_model


def make_trained_model_and_test_data():
    """Create a small trained model and holdout test set for evaluation tests."""

    messages = [
        "hello friend",
        "win free money",
        "meeting at noon",
        "claim your prize",
        "project update",
        "urgent cash offer",
    ]
    labels = ["ham", "spam", "ham", "spam", "ham", "spam"]

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(messages)

    X_train, X_test, y_train, y_test = split_training_data(
        features,
        labels,
        test_size=0.5,
        random_state=1,
    )

    model = create_model("logistic")
    train_model(model, X_train, y_train)

    return model, X_test, y_test


def test_evaluate_model_returns_expected_keys():
    """Test that evaluate_model returns all expected evaluation fields."""

    model, X_test, y_test = make_trained_model_and_test_data()

    evaluation = evaluate_model(model, X_test, y_test)

    assert set(evaluation.keys()) == {
        "accuracy",
        "confusion_matrix",
        "classes",
        "report",
    }


def test_evaluate_model_returns_accuracy_between_zero_and_one():
    """Test that the evaluation accuracy is a valid percentage value."""

    model, X_test, y_test = make_trained_model_and_test_data()

    evaluation = evaluate_model(model, X_test, y_test)

    assert 0.0 <= evaluation["accuracy"] <= 1.0


def test_evaluate_model_returns_model_classes():
    """Test that evaluation includes the labels learned by the trained model."""

    model, X_test, y_test = make_trained_model_and_test_data()

    evaluation = evaluate_model(model, X_test, y_test)

    assert set(evaluation["classes"]) == {"ham", "spam"}


def test_evaluate_model_returns_square_confusion_matrix():
    """Test that the confusion matrix has one row and column per class."""

    model, X_test, y_test = make_trained_model_and_test_data()

    evaluation = evaluate_model(model, X_test, y_test)
    confusion_matrix = evaluation["confusion_matrix"]
    class_count = len(evaluation["classes"])

    # A confusion matrix should be square: classes by classes.
    assert len(confusion_matrix) == class_count
    assert all(len(row) == class_count for row in confusion_matrix)


def test_evaluate_model_returns_classification_report_text():
    """Test that the classification report contains the learned class labels."""

    model, X_test, y_test = make_trained_model_and_test_data()

    evaluation = evaluate_model(model, X_test, y_test)
    report = evaluation["report"]

    assert isinstance(report, str)
    assert "ham" in report
    assert "spam" in report
