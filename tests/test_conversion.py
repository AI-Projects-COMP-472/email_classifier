import pytest

from src.conversion import SpamDataset, TextVectorizer


def test_spam_dataset_loads_valid_csv(tmp_path):
    """Test that SpamDataset.load reads a valid CSV and keeps usable rows."""

    csv_file = tmp_path / "spam.csv"
    csv_file.write_text(
        "label,message\n"
        "ham,Hello how are you?\n"
        "spam,Win money now!\n",
        encoding="utf-8",
    )

    dataset = SpamDataset.load(csv_file)

    # The loader should return the expected columns and rows.
    assert list(dataset.columns) == ["label", "message"]
    assert len(dataset) == 2

    # Values should be loaded correctly from the CSV file.
    assert dataset.loc[0, "label"] == "ham"
    assert dataset.loc[1, "message"] == "Win money now!"


def test_spam_dataset_removes_blank_or_missing_rows(tmp_path):
    """Test that rows with empty labels or messages are removed."""

    csv_file = tmp_path / "spam.csv"
    csv_file.write_text(
        "label,message\n"
        "ham,Hello there\n"
        "spam,\n"
        ",Missing label\n"
        "spam,Claim your prize\n",
        encoding="utf-8",
    )

    dataset = SpamDataset.load(csv_file)

    # Only rows with both a label and a message should remain.
    assert len(dataset) == 2
    assert dataset["label"].tolist() == ["ham", "spam"]
    assert dataset["message"].tolist() == ["Hello there", "Claim your prize"]


def test_spam_dataset_rejects_missing_required_columns(tmp_path):
    """Test that the loader raises an error if required columns are missing."""

    csv_file = tmp_path / "invalid.csv"
    csv_file.write_text(
        "category,text\n"
        "ham,Hello there\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        SpamDataset.load(csv_file)


def test_text_vectorizer_fit_transform_converts_text_to_features():
    """Test that TextVectorizer converts text messages into numeric features."""

    messages = [
        "hello friend",
        "win free money",
        "meeting with friend",
    ]

    vectorizer = TextVectorizer()
    feature_matrix = vectorizer.fit_transform(messages)

    # One feature row should be created for each input message.
    assert feature_matrix.shape[0] == len(messages)

    # The vectorizer should create at least one text feature.
    assert feature_matrix.shape[1] > 0


def test_text_vectorizer_transform_uses_existing_vocabulary():
    """Test that transform works after the vectorizer has been fitted."""

    training_messages = [
        "hello friend",
        "win free money",
    ]

    vectorizer = TextVectorizer()
    vectorizer.fit(training_messages)

    new_messages = ["hello money"]
    feature_matrix = vectorizer.transform(new_messages)

    # Transforming one new message should return one feature row.
    assert feature_matrix.shape[0] == 1

    # The number of columns should match the learned vocabulary size.
    assert feature_matrix.shape[1] == len(vectorizer.vectorizer.vocabulary_)