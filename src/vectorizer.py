"""
    Converting text to TF-IDF (term frequency-inverse document frequency) numerical features
    - TF (term frequency): How much the word appears in this specific message
    - IDF (inverse document frequency): How rare or common the word is across all training messages
    - gives higher importance to words that are useful for distinguishing one message from others
    - wrap scikit-learn’s TfidfVectorizer turning email text into machine-learning-ready numbers
"""

from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    """Wrapper for scikit-learn's TfidfVectorizer to convert text to numerical features.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) gives higher importance to words
    that are useful for distinguishing messages from each other.
    """
    
    def __init__(self, stop_words: str = "english", max_features: int = None):
        """Initialize the text vectorizer.

        Args:
            stop_words: Language to remove common words from (e.g., "the", "and", "is").
                Default is "english".
            max_features: Maximum number of word features to keep. None means no limit.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            max_features=max_features
        )

    def fit(self, messages):
        """Learn the vocabulary from training messages.
        
        This method learns which words exist in the training data and creates a mapping
        from words to feature columns. Must be called before transform().

        Args:
            messages: List or pandas Series of text messages to learn from.

        Returns:
            self: Allows method chaining (e.g., vectorizer.fit(x).transform(y)).
        """
        self.vectorizer.fit(messages)
        return self

    def transform(self, messages):
        """Convert messages to TF-IDF features using learned vocabulary.
        
        Transforms messages using the vocabulary learned from fit(). Does not learn
        from these messages - only applies the learned mapping.

        Args:
            messages: List or Series of text messages to transform.

        Returns:
            Sparse matrix of TF-IDF features. Sparse format saves memory since most
            messages contain only a small number of the total vocabulary words.
        """
        return self.vectorizer.transform(messages)

    def fit_transform(self, messages):
        """Learn vocabulary and transform messages in one step.
        
        Convenience method that calls fit() and then transform() in sequence.
        Used when transforming training data that will be used for learning.

        Args:
            messages: List or Series of text messages to fit and transform.

        Returns:
            Sparse matrix of TF-IDF features for the same messages.
        """
        return self.vectorizer.fit_transform(messages)