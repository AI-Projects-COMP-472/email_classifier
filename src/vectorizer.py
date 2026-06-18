"""
    Converting text to TF-IDF (term frequency-inverse document frequency) numerical features
    - gives higher importance to words that are useful for distinguishing one message from others
    - wrap scikit-learn’s TfidfVectorizer turning email text into machine-learning-ready numbers
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    # Constructor    
    def __init__(self, stop_words: str = "english", max_features: int = None):
        """
        Initialize the text vectorizer.

        Args:
            stop_words: removes common English words like “the”, “and”, “is”
            max_features: means there is no fixed limit on how many word features to keep
        """
        # Creates the actual scikit-learn TfidfVectorizer and stores it in self.vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            max_features=max_features
        )

    # Teaches the vectorizer the vocabulary from the training messages. 
    # It learns which words exist and how to map them to feature columns
    def fit(self, messages):
        """
        Fit the vectorizer on training messages.

        Args:
            messages: can be a list or pandas Series of text messages

        Returns:
            self for method chaining.
        """
        # learn vocabulary like: ["free", "win", "call", "meeting", "tomorrow"]
        self.vectorizer.fit(messages)
        # allows method chaining, like: vectorizer.fit(messages).transform(new_messages)
        return self

    def transform(self, messages):
        """
        Convert messages to TF-IDF features using vocabulary that was already learned with fit

        Args:
            messages: List or Series of text messages to transform.

        Returns:
            Sparse matrix of TF-IDF features 
            (most messages only contain a small number of all possible words, 
            so storing every zero would waste memory)
        """
        return self.vectorizer.transform(messages)

    def fit_transform(self, messages):
        """
        Fit the vectorizer and transform messages in one step (used on training data)

        Args:
            messages: List or Series of text messages to fit and transform.

        Returns:
            Sparse matrix of TF-IDF features.
        """
        return self.vectorizer.fit_transform(messages)