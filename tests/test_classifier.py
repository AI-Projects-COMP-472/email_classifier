import pandas as pd
import pytest

import src.classifier as classifier_module
from src.classifier import EmailClassifier


def make_sample_dataset():
    """Create a small balanced dataset for EmailClassifier tests."""

    return pd.DataFrame(
        {
            "label": [" ham ", "SPAM", "ham", "spam", "HAM", " spam "],
            "message": [
                "lets have lunch",
                "win money now",
                "meeting at noon",
                "free entry prize",
                "are you available",
                "urgent offer claim",
            ],
        }
    )


@pytest.fixture
def classifier_with_sample_dataset(monkeypatch):
    """Create an EmailClassifier that uses an in-memory dataset instead of a CSV."""

    sample = make_sample_dataset()

    monkeypatch.setattr(
        classifier_module.SpamDataset,
        "load",
        classmethod(lambda cls, path="data/spam.csv": sample.copy()),
    )

    return EmailClassifier()


def test_classifier_loads_and_normalizes_dataset(classifier_with_sample_dataset):
    """Test that EmailClassifier loads the dataset and normalizes labels."""

    classifier = classifier_with_sample_dataset

    # Labels should be lowercase and stripped of extra spaces.
    assert set(classifier.dataset["label"]) == {"ham", "spam"}
    assert len(classifier.dataset) == 6
    assert classifier.trained is False


def test_get_dataset_info_returns_summary(classifier_with_sample_dataset):
    """Test that dataset information includes useful summary details."""

    classifier = classifier_with_sample_dataset

    info = classifier.get_dataset_info()

    assert "Total records: 6" in info
    assert "Label distribution:" in info
    assert "Average message length:" in info


def test_train_sets_model_and_training_state(classifier_with_sample_dataset):
    """Test that train fits a model and stores the train/test split."""

    classifier = classifier_with_sample_dataset

    classifier.train(model_type="logistic", test_size=0.5, random_state=1)

    assert classifier.trained is True
    assert classifier.model is not None
    assert classifier.X_train is not None
    assert classifier.X_test is not None
    assert classifier.y_train is not None
    assert classifier.y_test is not None


def test_train_rejects_unsupported_model_type(classifier_with_sample_dataset):
    """Test that train raises an error for unsupported model types."""

    classifier = classifier_with_sample_dataset

    with pytest.raises(ValueError):
        classifier.train(model_type="unsupported")


def test_predict_requires_trained_classifier(classifier_with_sample_dataset):
    """Test that prediction is blocked before the classifier is trained."""

    classifier = classifier_with_sample_dataset

    with pytest.raises(ValueError):
        classifier.predict("free money offer")


def test_predict_rejects_empty_message(classifier_with_sample_dataset):
    """Test that prediction rejects blank input after training."""

    classifier = classifier_with_sample_dataset
    classifier.train(model_type="logistic", test_size=0.5, random_state=1)

    with pytest.raises(ValueError):
        classifier.predict("   ")


def test_predict_returns_label_and_confidence(classifier_with_sample_dataset):
    """Test that prediction returns a known label and confidence score."""

    classifier = classifier_with_sample_dataset
    classifier.train(model_type="logistic", test_size=0.5, random_state=1)

    label, confidence = classifier.predict("free money offer")

    assert label in {"ham", "spam"}
    assert 0.0 <= confidence <= 1.0


def test_evaluate_requires_trained_classifier(classifier_with_sample_dataset):
    """Test that evaluation is blocked before the classifier is trained."""

    classifier = classifier_with_sample_dataset

    with pytest.raises(ValueError):
        classifier.evaluate()


def test_evaluate_returns_model_metrics(classifier_with_sample_dataset):
    """Test that evaluation returns metrics after the classifier is trained."""

    classifier = classifier_with_sample_dataset
    classifier.train(model_type="logistic", test_size=0.5, random_state=1)

    evaluation = classifier.evaluate()

    assert "accuracy" in evaluation
    assert "confusion_matrix" in evaluation
    assert "classes" in evaluation
    assert "report" in evaluation
    assert 0.0 <= evaluation["accuracy"] <= 1.0
