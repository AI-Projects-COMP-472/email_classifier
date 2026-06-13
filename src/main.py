"""
COMP 472 Mini Project 2
AI-driven email filtering system for spam emails detection

Run:
    python src/main.py

Main entry point for the email classifier.
This module handles the command-line interface and user interaction.
"""

from src.classifier import EmailClassifier


def render_preview(classifier: EmailClassifier, rows: int = 5) -> None:
    preview = classifier.get_dataset_preview(rows)
    print("\n--- Dataset Preview ---")
    for index, row in preview.iterrows():
        print(f"[{index + 1}] {row['label'].upper()}: {row['message']}")
    print(f"{'---' * 8}\n")


def print_evaluation(evaluation: dict) -> None:
    classes = evaluation["classes"]
    confusion = evaluation["confusion_matrix"]
    print("--- Evaluation ---")
    print(f"Accuracy: {evaluation['accuracy']:.2%}")
    print("Confusion matrix:")
    print(f"  Classes: {classes}")
    for label, row in zip(classes, confusion):
        print(f"  {label}: {row}")
    print("\nClassification report:")
    print(evaluation["report"])
    print(f"{'---' * 8}\n")


def run_prediction_loop(classifier: EmailClassifier) -> None:
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

    try:
        classifier = EmailClassifier(knowledge_base_path="data/spam.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\n--- Dataset Information ---")
    print(classifier.get_dataset_info())
    render_preview(classifier)

    try:
        classifier.train(model_type="logistic")
    except ValueError as error:
        print(f"Error while training: {error}")
        return

    evaluation = classifier.evaluate()
    print_evaluation(evaluation)

    run_prediction_loop(classifier)


if __name__ == "__main__":
    main()
