from src.vectorizer import TextVectorizer


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