"""Data loading and validation for the email classifier.

Used for:
- Reading the CSV file into a DataFrame
- Verifying required columns exist
- Cleaning rows -> drop missing or blank values
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

class SpamDataset:
    """Loads and validates the CSV spam dataset."""

    # Columns required by the rest of the classifier pipeline
    required_columns = {"label", "message"}

    @classmethod
    def load(cls, file_path: Path | str = "data/spam.csv") -> pd.DataFrame:
        """
        Load the spam dataset CSV file using pandas.

        Args:
            file_path: Path to the CSV file. Default: 'data/spam.csv'

        Returns:
            A cleaned DataFrame with label and message columns.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is missing required columns or valid rows.
        """
        # Convert to Path object for cross-platform compatibility
        file_path = Path(file_path)

        # Fail fast if the CSV is missing
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find {file_path}. Make sure spam.csv is in the data/ folder."
            )

        # Read raw data with error handling for different encodings
        try:
            # Try UTF-8 first (most common)
            data = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, try latin-1 (common for some datasets)
            try:
                data = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                # If latin-1 fails, try iso-8859-1
                data = pd.read_csv(file_path, encoding='iso-8859-1')

        # Validate schema -> must have label + message columns
        if not cls.required_columns.issubset(data.columns):
            raise ValueError(
                f"spam.csv must contain these required columns: {cls.required_columns}. "
                f"Found: {set(data.columns)}"
            )

        # Drop rows missing either field
        data = data.dropna(subset=["label", "message"]).copy()
        data["label"] = data["label"].astype(str).str.strip()
        data["message"] = data["message"].astype(str).str.strip()

        # Remove empty strings after stripping
        data = data[(data["label"] != "") & (data["message"] != "")]

        # Ensure there's at least one usable row
        if data.empty:
            raise ValueError("spam.csv does not contain any valid label-message rows.")

        # Return a clean, sequential index
        return data.reset_index(drop=True)
    
class TextVectorizer:
    """Converts text messages to TF-IDF numerical features."""

    def __init__(self, stop_words: str = "english", max_features: int = None):
        """
        Initialize the text vectorizer.

        Args:
            stop_words: Language for stop words removal (e.g., 'english').
            max_features: Maximum number of features to extract. None for unlimited.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            max_features=max_features
        )

    def fit(self, messages):
        """
        Fit the vectorizer on training messages.

        Args:
            messages: List or Series of text messages to fit on.

        Returns:
            self for method chaining.
        """
        self.vectorizer.fit(messages)
        return self

    def transform(self, messages):
        """
        Convert messages to TF-IDF features.

        Args:
            messages: List or Series of text messages to transform.

        Returns:
            Sparse matrix of TF-IDF features.
        """
        return self.vectorizer.transform(messages)

    def fit_transform(self, messages):
        """
        Fit the vectorizer and transform messages in one step.

        Args:
            messages: List or Series of text messages to fit and transform.

        Returns:
            Sparse matrix of TF-IDF features.
        """
        return self.vectorizer.fit_transform(messages)