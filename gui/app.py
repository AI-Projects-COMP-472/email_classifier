
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

from src.classifier import EmailClassifier

# gui/app.py is inside the gui/ folder, parents[1] points back to
# the project root: email_classifier/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "spam.csv"

class EmailClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Spam Classifier")
        self.root.geometry("600x450")

        self.classifier = None

        self.build_interface()
        self.load_model()

    def build_interface(self):
        title = tk.Label(
            self.root,
            text="Email Spam Classifier",
            font=("Arial", 18, "bold"),
        )
        title.pack(pady=10)

        self.status_label = tk.Label(
            self.root,
            text="Loading model...",
            font=("Arial", 10),
        )
        self.status_label.pack()

        input_label = tk.Label(
            self.root,
            text="Enter an email message:",
            font=("Arial", 11),
        )
        input_label.pack(anchor="w", padx=20, pady=(15, 5))

        self.message_input = tk.Text(
            self.root,
            height=8,
            width=65,
            wrap="word",
        )
        self.message_input.pack(padx=20)

        classify_button = tk.Button(
            self.root,
            text="Classify",
            command=self.classify_message,
            width=15,
        )
        classify_button.pack(pady=15)

        self.result_label = tk.Label(
            self.root,
            text="Prediction will appear here.",
            font=("Arial", 13, "bold"),
        )
        self.result_label.pack(pady=5)

        self.confidence_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 11),
        )
        self.confidence_label.pack()

        self.info_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 9),
            justify="left",
        )
        self.info_label.pack(padx=20, pady=20, anchor="w")

    def load_model(self):
        try:
            self.classifier = EmailClassifier(
                dataset_path=DATASET_PATH,
                spam_threshold=0.35,
            )
            self.classifier.train(model_type="logistic")

            self.status_label.config(text="Model ready")
            self.info_label.config(text=self.classifier.get_dataset_info())

        except (FileNotFoundError, ValueError) as error:
            self.status_label.config(text="Model failed to load")
            messagebox.showerror("Error", str(error))

    def classify_message(self):
        if self.classifier is None:
            messagebox.showerror("Error", "The model is not ready.")
            return

        message = self.message_input.get("1.0", tk.END).strip()

        if not message:
            messagebox.showwarning("Missing message", "Please enter an email message.")
            return

        try:
            label, confidence = self.classifier.predict(message)

            if label == "spam":
                result_text = "Prediction: SPAM"
                result_color = "red"
            else:
                result_text = "Prediction: HAM"
                result_color = "green"

            self.result_label.config(
                text=result_text,
                fg=result_color,
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2%}",
            )

        except ValueError as error:
            messagebox.showerror("Error", str(error))


def main():
    root = tk.Tk()
    app = EmailClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()