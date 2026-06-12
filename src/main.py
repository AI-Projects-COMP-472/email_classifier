"""
COMP 472 Mini Project 2
AI-driven email filtering system for spam emails detection

Run:
    python src/main.py

Main entry point for the email classifier.
This module handles the command-line interface and user interaction.
"""

from classifier import EmailClassifier


def main() -> None:
    """Main function to run the email classifier."""
    print("=" * 50)
    print("  Email Classifier - COMP 472")
    print("=" * 50)

    # Initialize classifier and load dataset
    try:
        classifier = EmailClassifier(knowledge_base_path="data/spam.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    # Display dataset information
    print(f"\n--- Dataset Information ---")
    print(classifier.get_dataset_info())
    print(f"{'---' * 8}\n")

    # TODO: Display preview of first few messages
    # TODO: Add text vectorization
    # TODO: Add model training
    # TODO: Add model evaluation
    # TODO: Add interactive prediction loop


if __name__ == "__main__":
    main()