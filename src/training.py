from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def create_model(model_type: str):
    """Create a supported machine learning model."""

    if model_type.lower() in {"logistic", "lr"}:
        return LogisticRegression(max_iter=1000)

    if model_type.lower() in {"naive_bayes", "nb", "multinomial_nb"}:
        return MultinomialNB()

    raise ValueError("Unsupported model type. Choose 'logistic' or 'naive_bayes'.")


def split_training_data(features, labels, test_size=0.2, random_state=42):
    """Split feature and label data into training and testing sets."""

    return train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )


def train_model(model, X_train, y_train):
    """Train a model using the provided training data."""

    model.fit(X_train, y_train)
    return model