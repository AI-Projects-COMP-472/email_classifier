
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import EmailClassifier


DATASET_PATH = PROJECT_ROOT / "data" / "spam.csv"

MODEL_THRESHOLDS = {
    "logistic": 0.35,
    "naive_bayes": 0.20,
}


class EmailClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("COMP 472 - Mini Project 2")
        self.root.geometry("1100x760")
        self.root.minsize(950, 820)

        self.classifier = None
        self.chart_canvas = None
        self.model_type = tk.StringVar(value="logistic")

        self.configure_styles()
        self.build_interface()
        self.load_model()

    def configure_styles(self):
        self.root.configure(bg="#2f2f2f")

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "App.TFrame",
            background="#2f2f2f", #this main bg
        )
        style.configure(
            "Panel.TLabelframe",
            background="#2f2f2f",
            foreground="#ffffff",
            bordercolor="#555555",
            relief="solid",
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background="#2f2f2f",
            foreground="#ffffff",
            font=("Arial", 12, "bold"),
        )
        style.configure(
            "App.TLabel",
            background="#2f2f2f",
            foreground="#ffffff",
            font=("Arial", 11),
        )
        style.configure(
            "Panel.TLabel",
            background="#2f2f2f",
            foreground="#ffffff",
            font=("Arial", 11),
        )
        style.configure(
            "Muted.TLabel",
            background="#3a3a3a",
            foreground="#cfcfcf",
            font=("Arial", 10),
        )
        style.configure(
            "Title.TLabel",
            background="#2f2f2f",
            foreground="#ffffff",
            font=("Arial", 26, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background="#454545",
            foreground="#ffffff",
            padding=(12, 5),
            font=("Arial", 10, "bold"),
        )
        style.configure(
            "Primary.TButton",
            font=("Arial", 11, "bold"),
            padding=(16, 8),
        )
        style.configure(
            "Secondary.TButton",
            font=("Arial", 11),
            padding=(16, 8),
        )
        style.configure(
            "Treeview",
            rowheight=28,
            font=("Arial", 10),
            background="#f5f5f5",
            fieldbackground="#f5f5f5",
            foreground="#111111",
        )
        style.configure(
            "Treeview.Heading",
            font=("Arial", 10, "bold"),
        )

    def build_interface(self):
        main_container = ttk.Frame(self.root, style="App.TFrame", padding=24)
        main_container.pack(fill="both", expand=True)

        header = ttk.Frame(main_container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 18))

        title = ttk.Label(
            header,
            text="Email Spam Classifier",
            style="Title.TLabel",
        )
        title.pack(anchor="center")

        self.status_label = ttk.Label(
            header,
            text="Loading model...",
            style="Status.TLabel",
        )
        self.status_label.pack(anchor="center", pady=(10, 14))

        model_row = ttk.Frame(header, style="App.TFrame")
        model_row.pack(anchor="center")

        model_label = ttk.Label(
            model_row,
            text="Model",
            style="App.TLabel",
        )
        model_label.pack(side="left", padx=(0, 8))

        self.model_selector = ttk.Combobox(
            model_row,
            textvariable=self.model_type,
            values=["logistic", "naive_bayes"],
            state="readonly",
            width=18,
        )
        self.model_selector.pack(side="left")
        self.model_selector.bind("<<ComboboxSelected>>", self.change_model)

        content = ttk.Frame(main_container, style="App.TFrame")
        content.pack(fill="both", expand=True)

        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        left_column = ttk.Frame(content, style="App.TFrame")
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        right_column = ttk.Frame(content, style="App.TFrame")
        right_column.grid(row=0, column=1, sticky="nsew", padx=(14, 0))

        self.build_message_panel(left_column)
        self.build_result_panel(left_column)

        self.build_dataset_panel(right_column)
        self.build_evaluation_panel(right_column)
        self.build_chart_panel(right_column)

    def build_message_panel(self, parent):
        message_panel = ttk.LabelFrame(
            parent,
            text="Message",
            style="Panel.TLabelframe",
            padding=16,
        )
        message_panel.pack(fill="both", expand=True)

        input_label = ttk.Label(
            message_panel,
            text="Enter an email message:",
            style="Panel.TLabel",
        )
        input_label.pack(anchor="w", pady=(0, 8))

        self.message_input = tk.Text(
            message_panel,
            height=12,
            wrap="word",
            bg="#181818",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief="flat",
            padx=12,
            pady=12,
            font=("Arial", 11),
        )
        self.message_input.pack(fill="both", expand=True)

        button_row = ttk.Frame(message_panel, style="Panel.TLabelframe")
        button_row.pack(fill="x", pady=(16, 0))

        classify_button = ttk.Button(
            button_row,
            text="Classify",
            command=self.classify_message,
            style="Primary.TButton",
        )
        classify_button.pack(side="left", fill="x", expand=True, padx=(0, 8))

        clear_button = ttk.Button(
            button_row,
            text="Clear",
            command=self.clear_message,
            style="Secondary.TButton",
        )
        clear_button.pack(side="left", fill="x", expand=True, padx=(8, 0))

    def build_result_panel(self, parent):
        result_panel = ttk.LabelFrame(
            parent,
            text="Prediction",
            style="Panel.TLabelframe",
            padding=16,
        )
        result_panel.pack(fill="x", pady=(18, 0))

        self.result_label = ttk.Label(
            result_panel,
            text="Prediction will appear here.",
            style="Panel.TLabel",
            font=("Arial", 16, "bold"),
        )
        self.result_label.pack(anchor="w")

        self.confidence_label = ttk.Label(
            result_panel,
            text="Confidence: --",
            style="Muted.TLabel",
        )
        self.confidence_label.pack(anchor="w", pady=(6, 0))

    def build_dataset_panel(self, parent):
        dataset_panel = ttk.LabelFrame(
            parent,
            text="Dataset",
            style="Panel.TLabelframe",
            padding=16,
        )
        dataset_panel.pack(fill="x")

        self.total_records_label = ttk.Label(
            dataset_panel,
            text="Total records: --",
            style="Panel.TLabel",
        )
        self.total_records_label.pack(anchor="w", pady=2)

        self.ham_count_label = ttk.Label(
            dataset_panel,
            text="Ham messages: --",
            style="Panel.TLabel",
        )
        self.ham_count_label.pack(anchor="w", pady=2)

        self.spam_count_label = ttk.Label(
            dataset_panel,
            text="Spam messages: --",
            style="Panel.TLabel",
        )
        self.spam_count_label.pack(anchor="w", pady=2)

        self.avg_length_label = ttk.Label(
            dataset_panel,
            text="Average length: --",
            style="Panel.TLabel",
        )
        self.avg_length_label.pack(anchor="w", pady=2)

    def build_evaluation_panel(self, parent):
        evaluation_panel = ttk.LabelFrame(
            parent,
            text="Evaluation",
            style="Panel.TLabelframe",
            padding=16,
        )
        evaluation_panel.pack(fill="x", pady=(18, 0))

        self.accuracy_label = ttk.Label(
            evaluation_panel,
            text="Accuracy: --",
            style="Panel.TLabel",
            font=("Arial", 12, "bold"),
        )
        self.accuracy_label.pack(anchor="w", pady=(0, 10))

        self.confusion_table = ttk.Treeview(
            evaluation_panel,
            columns=("predicted_ham", "predicted_spam"),
            show="headings",
            height=2,
        )
        self.confusion_table.heading("predicted_ham", text="Predicted ham")
        self.confusion_table.heading("predicted_spam", text="Predicted spam")
        self.confusion_table.column("predicted_ham", anchor="center", width=120)
        self.confusion_table.column("predicted_spam", anchor="center", width=120)
        self.confusion_table.pack(fill="x")

    def build_chart_panel(self, parent):
        chart_panel = ttk.LabelFrame(
            parent,
            text="Label Distribution",
            style="Panel.TLabelframe",
            padding=12,
        )
        chart_panel.pack(fill="both", expand=True, pady=(18, 0))

        self.chart_frame = ttk.Frame(chart_panel, style="Panel.TLabelframe")
        self.chart_frame.pack(fill="both", expand=True)

    def load_model(self):
        try:
            model_type = self.model_type.get()
            spam_threshold = MODEL_THRESHOLDS[model_type]

            self.status_label.config(text=f"Training model: {model_type}")
            self.root.update_idletasks()

            self.classifier = EmailClassifier(
                dataset_path=DATASET_PATH,
                spam_threshold=spam_threshold,
            )
            self.classifier.train(model_type=model_type)

            evaluation = self.classifier.evaluate()

            self.status_label.config(
                text=f"Model ready: {model_type} | Threshold: {spam_threshold:.2f}"
            )

            self.update_dataset_summary()
            self.update_evaluation_summary(evaluation)
            self.display_label_distribution_chart()
            self.clear_message()

        except (FileNotFoundError, ValueError) as error:
            self.status_label.config(text="Model failed to load")
            messagebox.showerror("Error", str(error))

    def update_dataset_summary(self):
        dataset = self.classifier.dataset
        label_counts = dataset["label"].value_counts().to_dict()
        avg_length = dataset["message"].str.len().mean()

        self.total_records_label.config(text=f"Total records: {len(dataset)}")
        self.ham_count_label.config(text=f"Ham messages: {label_counts.get('ham', 0)}")
        self.spam_count_label.config(text=f"Spam messages: {label_counts.get('spam', 0)}")
        self.avg_length_label.config(
            text=f"Average length: {avg_length:.0f} characters"
        )

    def update_evaluation_summary(self, evaluation):
        self.accuracy_label.config(
            text=f"Accuracy: {evaluation['accuracy']:.2%}"
        )

        for item in self.confusion_table.get_children():
            self.confusion_table.delete(item)

        classes = [str(label) for label in evaluation["classes"]]
        confusion = evaluation["confusion_matrix"]

        for label, row in zip(classes, confusion):
            self.confusion_table.insert(
                "",
                "end",
                text=f"Actual {label}",
                values=(row[0], row[1]),
            )

        self.confusion_table.configure(show="tree headings")
        self.confusion_table.heading("#0", text="Actual")
        self.confusion_table.column("#0", width=90, anchor="w")

    def display_label_distribution_chart(self):
        label_counts = (
            self.classifier.dataset["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
        )

        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        figure = Figure(figsize=(4.0, 2.6), dpi=100)
        figure.patch.set_facecolor("#757575")
        


        axis = figure.add_subplot(111)
        axis.set_facecolor("#ffffff")

        labels = list(label_counts.index)
        values = list(label_counts.values)
        colors = ["#4C78A8", "#F58518"]

        bars = axis.bar(labels, values, color=colors[:len(labels)])

        axis.set_title("Message Label Distribution", fontsize=9)
        axis.set_xlabel("Label", fontsize=8)
        axis.set_ylabel("Messages", fontsize=8)
        axis.tick_params(axis="x", labelsize=8, rotation=0)
        axis.tick_params(axis="y", labelsize=8)

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 0.92,
                str(value),
                ha="center",
                va="top",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

        figure.tight_layout()

        self.chart_canvas = FigureCanvasTkAgg(figure, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

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
                result_text = "SPAM"
                result_color = "#ff6b6b"
            else:
                result_text = "HAM"
                result_color = "#51cf66"

            self.result_label.config(
                text=f"Prediction: {result_text}",
                foreground=result_color,
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
            foreground="#ffffff",
        )
        self.confidence_label.config(text="Confidence: --")

    def change_model(self, event=None):
        self.load_model()


def main():
    root = tk.Tk()
    EmailClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()