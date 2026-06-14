
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.classifier import EmailClassifier

# gui/app.py is inside the gui/ folder, parents[1] points back to
# the project root: email_classifier/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "spam.csv"

class EmailClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("COMP 472 - Mini Project 2")
        self.root.geometry("600x800")

        self.classifier = None

        self.model_type = tk.StringVar(value="logistic")
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

        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=(10, 0))

        model_label = tk.Label(
            model_frame,
            text="Model:",
            font=("Arial", 10),
        )
        model_label.pack(side="left", padx=(0, 8))

        self.model_selector = ttk.Combobox(
            model_frame,
            textvariable=self.model_type,
            values=["logistic", "naive_bayes"],
            state="readonly",
            width=15,
        )
        self.model_selector.pack(side="left")

        self.model_selector.bind("<<ComboboxSelected>>", self.change_model)

        self.message_input = tk.Text(
            self.root,
            height=8,
            width=65,
            wrap="word",
        )
        self.message_input.pack(padx=20)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=15)

        classify_button = tk.Button(
            button_frame,
            text="Classify",
            command=self.classify_message,
            width=15,
        )
        classify_button.pack(side="left", padx=5)

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_message,
            width=15,
        )
        clear_button.pack(side="left", padx=5)

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

        self.evaluation_label = tk.Label(
            self.root,
            text="",
            font=("Courier New", 10),
            justify="left",
        )
        self.evaluation_label.pack(padx=20, pady=(0, 15), anchor="w")

        self.chart_frame = tk.Frame(self.root)
        self.chart_frame.pack(padx=20, pady=10, fill="both")

    def load_model(self):

        thresholds = {
            "logistic": 0.35,
            "naive_bayes": 0.20,
        }

        try:
            self.status_label.config(text="Training model...")
            self.root.update_idletasks()

            self.classifier = EmailClassifier(
                dataset_path=DATASET_PATH,
                spam_threshold=thresholds[self.model_type.get()],
            )
            self.classifier.train(model_type=self.model_type.get())

            evaluation = self.classifier.evaluate()
            evaluation_text = self.format_evaluation_summary(evaluation)

            self.status_label.config(
                text=f"Model ready: {self.model_type.get()}"
            )
            self.info_label.config(text=self.classifier.get_dataset_info())
            self.evaluation_label.config(text=evaluation_text)

            self.display_label_distribution_chart()

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

    def clear_message(self):
        self.message_input.delete("1.0", tk.END)
        self.result_label.config(
            text="Prediction will appear here.",
            fg="black",
        )
        self.confidence_label.config(text="")

    def change_model(self, event=None):
        self.result_label.config(
            text="Prediction will appear here.",
            fg="black",
        )
        self.confidence_label.config(text="")
        self.load_model()
    
    def format_evaluation_summary(self, evaluation):
        classes = [str(label) for label in evaluation["classes"]]
        confusion = evaluation["confusion_matrix"]

        header = f"{'Actual':<10}{'Predicted ham':>15}{'Predicted spam':>17}"
        separator = "-" * len(header)

        lines = [
            "--- Evaluation ---",
            f"Accuracy: {evaluation['accuracy']:.2%}",
            "",
            "Confusion matrix:",
            header,
            separator,
        ]

        for label, row in zip(classes, confusion):
            lines.append(
                f"{label:<10}{row[0]:>15}{row[1]:>17}"
            )

        return "\n".join(lines)
    
    def display_label_distribution_chart(self):
        if self.classifier is None:
            return

        label_counts = (
            self.classifier.dataset["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
        )

        figure = Figure(figsize=(3.5, 2.5), dpi=100)
        axis = figure.add_subplot(111)

        label_counts.plot(
            kind="bar",
            ax=axis,
            color=["#4C78A8", "#F58518"],
        )

        axis.set_title("Message Label Distribution")
        axis.set_xlabel("Label")
        axis.set_ylabel("Messages")
        axis.tick_params(axis="x", rotation=0)

        for index, count in enumerate(label_counts):
            axis.text(index, count, str(count), ha="center", va="bottom")

        figure.tight_layout()

        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(figure, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


def main():
    root = tk.Tk()
    app = EmailClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()