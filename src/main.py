"""Command-line interface for the COMP 472 email classifier.

This module provides an interactive workflow for training a spam detection
model, evaluating its performance, generating a visualization of label
distribution, and classifying new email messages entered by the user.

Run:
    python -m src.main
"""

from src.classifier import EmailClassifier
from src.visualization import plot_label_distribution

MODEL_THRESHOLDS = {
    "logistic": 0.35,
    "naive_bayes": 0.20,
}


def choose_model() -> str:
    """Prompt the user to select the classifier model.

    Returns:
        A model type string of either 'logistic' or 'naive_bayes'.
    """

    choices = {
        "1": "logistic",
        "logistic": "logistic",
        "lr": "logistic",
        "2": "naive_bayes",
        "naive_bayes": "naive_bayes",
        "naive bayes": "naive_bayes",
        "nb": "naive_bayes",
    }

    print("\nChoose a model:")
    print("  1. Logistic Regression")
    print("  2. Naive Bayes")

    while True:
        user_choice = input("Model [1]: ").strip().lower()

        if not user_choice:
            return "logistic"

        if user_choice in choices:
            return choices[user_choice]

        print("Invalid choice. Enter 1 for logistic or 2 for naive_bayes.")


def print_evaluation(evaluation: dict) -> None:
    """Print evaluation metrics for the trained classifier.

    Args:
        evaluation: A dictionary returned by EmailClassifier.evaluate(),
            containing accuracy, confusion matrix, labels, and report text.
    """
    classes = evaluation["classes"]
    confusion = evaluation["confusion_matrix"]
    print("\n--- Evaluation ---")
    print(f"Accuracy: {evaluation['accuracy']:.2%}")
    print("Confusion matrix:")
    print(f"  Classes: {classes}")
    for label, row in zip(classes, confusion):
        print(f"  {label}: {row}")
    print(f"{'---' * 8}\n")


def run_prediction_loop(classifier: EmailClassifier) -> None:
    """Run the interactive prediction loop for user-entered email text.

    The loop continues until the user types 'quit', 'exit', or 'q'.
    """
    print("Enter an email message to classify, or type 'quit' to exit.")
    while True:
        user_input = input("Message: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            print("Please enter some text or type 'quit' to exit.")
            continue

        try:
            label, confidence = classifier.predict(user_input)
            print(f"Prediction: {label.upper()} | Confidence: {confidence:.2%}\n")
        except ValueError as error:
            print(f"Error: {error}\n")


def main() -> None:
    """Main function to run the email classifier."""
    print("=" * 50)
    print("  Email Classifier - COMP 472")
    print("=" * 50)

    model_type = choose_model()
    spam_threshold = MODEL_THRESHOLDS[model_type]

    try:
        classifier = EmailClassifier(
            dataset_path="data/spam.csv",
            spam_threshold=spam_threshold,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\n--- Dataset Information ---")
    print(classifier.get_dataset_info())
    print(f"Model: {model_type}")
    print(f"Spam threshold: {spam_threshold:.2f}")

    try:
        classifier.train(model_type=model_type)

    except ValueError as error:
        print(f"Error while training: {error}")
        return

    evaluation = classifier.evaluate()
    print_evaluation(evaluation)

    # Generate visualization
    plot_label_distribution(classifier.dataset)

    run_prediction_loop(classifier)


if __name__ == "__main__":
    main()
