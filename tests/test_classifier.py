import pytest
import pandas as pd

import src.classifier as classifier_module
from src.classifier import EmailClassifier


def make_sample_dataset():
    return pd.DataFrame(
        {
            "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
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


def test_train_predict_and_evaluate(monkeypatch):
    sample = make_sample_dataset()
    monkeypatch.setattr(
        classifier_module.SpamDataset,
        "load",
        classmethod(lambda cls, path="data/spam.csv": sample),
    )

    classifier = EmailClassifier()
    classifier.train(model_type="logistic", test_size=0.5, random_state=1)

    assert classifier.trained is True
    assert classifier.model is not None

    label, confidence = classifier.predict("free money offer")
    assert label in {"ham", "spam"}
    assert 0.0 <= confidence <= 1.0

    evaluation = classifier.evaluate()
    assert "accuracy" in evaluation
    assert 0.0 <= evaluation["accuracy"] <= 1.0
    assert evaluation["confusion_matrix"] is not None


def test_train_invalid_model_type(monkeypatch):
    sample = make_sample_dataset()
    monkeypatch.setattr(
        classifier_module.SpamDataset,
        "load",
        classmethod(lambda cls, path="data/spam.csv": sample),
    )

    classifier = EmailClassifier()

    with pytest.raises(ValueError):
        classifier.train(model_type="unsupported")


def test_predict_without_training(monkeypatch):
    sample = make_sample_dataset()
    monkeypatch.setattr(
        classifier_module.SpamDataset,
        "load",
        classmethod(lambda cls, path="data/spam.csv": sample),
    )

    classifier = EmailClassifier()

    with pytest.raises(ValueError):
        classifier.predict("hello world")


def test_evaluate_without_training(monkeypatch):
    sample = make_sample_dataset()
    monkeypatch.setattr(
        classifier_module.SpamDataset,
        "load",
        classmethod(lambda cls, path="data/spam.csv": sample),
    )

    classifier = EmailClassifier()

    with pytest.raises(ValueError):
        classifier.evaluate()


def test_train_naive_bayes(monkeypatch):
    from sklearn.naive_bayes import MultinomialNB

    sample = make_sample_dataset()
    monkeypatch.setattr(
        classifier_module.SpamDataset,
        "load",
        classmethod(lambda cls, path="data/spam.csv": sample),
    )

    classifier = EmailClassifier()
    classifier.train(model_type="naive_bayes", test_size=0.5, random_state=1)

    assert classifier.trained
    assert isinstance(classifier.model, MultinomialNB)


def test_predict_probabilities_sum(monkeypatch):
    sample = make_sample_dataset()
    monkeypatch.setattr(
        classifier_module.SpamDataset,
        "load",
        classmethod(lambda cls, path="data/spam.csv": sample),
    )

    classifier = EmailClassifier()
    classifier.train(model_type="logistic", test_size=0.5, random_state=1)

    matrix = classifier.vectorizer.transform(["test message probability"])
    probs = classifier.model.predict_proba(matrix)[0]
    assert abs(sum(probs) - 1.0) < 1e-6
