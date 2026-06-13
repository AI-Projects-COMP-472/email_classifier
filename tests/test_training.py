import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.training import create_model, split_training_data, train_model


def make_sample_features_and_labels():
    """Create a small balanced dataset for training tests."""

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

    return features, labels


def test_create_model_returns_logistic_regression():
    """Test that the logistic model option creates a LogisticRegression model."""

    model = create_model("logistic")

    assert isinstance(model, LogisticRegression)


def test_create_model_returns_naive_bayes():
    """Test that the naive_bayes model option creates a MultinomialNB model."""

    model = create_model("naive_bayes")

    assert isinstance(model, MultinomialNB)


def test_create_model_rejects_unsupported_model_type():
    """Test that an unsupported model type raises a clear ValueError."""

    with pytest.raises(ValueError):
        create_model("unsupported")


def test_split_training_data_creates_train_and_test_sets():
    """Test that features and labels are split into matching train/test sets."""

    features, labels = make_sample_features_and_labels()

    X_train, X_test, y_train, y_test = split_training_data(
        features,
        labels,
        test_size=0.5,
        random_state=1,
    )

    # The split should preserve matching row counts between features and labels.
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)

    # With six records and test_size=0.5, each set should contain three records.
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 3


def test_train_model_fits_and_returns_model():
    """Test that train_model fits the provided model and returns it."""

    features, labels = make_sample_features_and_labels()
    X_train, _, y_train, _ = split_training_data(
        features,
        labels,
        test_size=0.5,
        random_state=1,
    )
    model = create_model("logistic")

    trained_model = train_model(model, X_train, y_train)

    # A fitted scikit-learn classifier should learn its class labels.
    assert trained_model is model
    assert set(trained_model.classes_) == {"ham", "spam"}
