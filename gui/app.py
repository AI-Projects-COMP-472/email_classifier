
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
        """Configure ttk styles for a modern, clean appearance."""
        # Color palette
        BG_DARK = "#1e1e2e"      # Main background - darker
        BG_PANEL = "#2a2a3e"     # Panel background
        BG_INPUT = "#0f0f1e"     # Input background - darker
        TEXT_PRIMARY = "#ffffff"  # Primary text
        TEXT_MUTED = "#b0b0b0"   # Muted text
        BORDER_COLOR = "#3a3a4e" # Border color
        ACCENT_BLUE = "#5b8ef7"  # Primary accent blue
        ACCENT_GREEN = "#51cf66" # Success/ham color
        ACCENT_RED = "#ff6b6b"   # Alert/spam color

        self.root.configure(bg=BG_DARK)

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "App.TFrame",
            background=BG_DARK,
        )
        style.configure(
            "Panel.TLabelframe",
            background=BG_PANEL,
            foreground=TEXT_PRIMARY,
            bordercolor=BORDER_COLOR,
            relief="solid",
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background=BG_PANEL,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "App.TLabel",
            background=BG_DARK,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Panel.TLabel",
            background=BG_PANEL,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background=BG_PANEL,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Title.TLabel",
            background=BG_DARK,
            foreground=TEXT_PRIMARY,
            font=("Segoe UI", 28, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=BORDER_COLOR,
            foreground=TEXT_MUTED,
            padding=(16, 8),
            font=("Segoe UI", 9),
        )
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 8),
            background=ACCENT_BLUE,
            foreground=TEXT_PRIMARY,
        )
        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            padding=(14, 8),
        )
        style.configure(
            "Treeview",
            rowheight=26,
            font=("Segoe UI", 9),
            background="#f0f0f0",
            fieldbackground="#f0f0f0",
            foreground="#1e1e2e",
        )
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 9, "bold"),
            background="#e8e8e8",
            foreground="#1e1e2e",
        )
        style.map("Treeview", background=[("selected", ACCENT_BLUE)])

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
        """Build the message input panel with classification controls."""
        message_panel = ttk.LabelFrame(
            parent,
            text="Email Message",
            style="Panel.TLabelframe",
            padding=16,
        )
        message_panel.pack(fill="both", expand=True)

        # Input instructions
        input_label = ttk.Label(
            message_panel,
            text="Paste or type an email message below:",
            style="Panel.TLabel",
            font=("Segoe UI", 10),
        )
        input_label.pack(anchor="w", pady=(0, 10))

        # Text input area
        self.message_input = tk.Text(
            message_panel,
            height=12,
            wrap="word",
            bg="#0f0f1e",
            fg="#ffffff",
            insertbackground="#5b8ef7",
            relief="solid",
            borderwidth=1,
            padx=12,
            pady=12,
            font=("Segoe UI", 10),
        )
        self.message_input.pack(fill="both", expand=True)

        # Button row
        button_row = ttk.Frame(message_panel, style="Panel.TLabelframe")
        button_row.pack(fill="x", pady=(16, 0))

        classify_button = ttk.Button(
            button_row,
            text="🔍 Classify",
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
        """Build the prediction results display panel."""
        result_panel = ttk.LabelFrame(
            parent,
            text="Prediction Result",
            style="Panel.TLabelframe",
            padding=20,
        )
        result_panel.pack(fill="x", pady=(20, 0))

        # Prediction label - large and bold
        self.result_label = ttk.Label(
            result_panel,
            text="Awaiting classification...",
            style="Panel.TLabel",
            font=("Segoe UI", 20, "bold"),
        )
        self.result_label.pack(anchor="w", pady=(0, 12))

        # Confidence score
        self.confidence_label = ttk.Label(
            result_panel,
            text="Confidence: --",
            style="Muted.TLabel",
            font=("Segoe UI", 11),
        )
        self.confidence_label.pack(anchor="w")

    def build_dataset_panel(self, parent):
        """Build the dataset statistics panel."""
        dataset_panel = ttk.LabelFrame(
            parent,
            text="Dataset Statistics",
            style="Panel.TLabelframe",
            padding=16,
        )
        dataset_panel.pack(fill="x")

        self.total_records_label = ttk.Label(
            dataset_panel,
            text="Total records: --",
            style="Panel.TLabel",
        )
        self.total_records_label.pack(anchor="w", pady=3)

        self.ham_count_label = ttk.Label(
            dataset_panel,
            text="✓ Ham messages: --",
            style="Panel.TLabel",
        )
        self.ham_count_label.pack(anchor="w", pady=3)

        self.spam_count_label = ttk.Label(
            dataset_panel,
            text="⚠ Spam messages: --",
            style="Panel.TLabel",
        )
        self.spam_count_label.pack(anchor="w", pady=3)

        self.avg_length_label = ttk.Label(
            dataset_panel,
            text="Average length: --",
            style="Panel.TLabel",
        )
        self.avg_length_label.pack(anchor="w", pady=3)

    def build_evaluation_panel(self, parent):
        """Build the model evaluation metrics panel."""
        evaluation_panel = ttk.LabelFrame(
            parent,
            text="Model Evaluation",
            style="Panel.TLabelframe",
            padding=16,
        )
        evaluation_panel.pack(fill="x", pady=(18, 0))

        # Accuracy score - prominent display
        self.accuracy_label = ttk.Label(
            evaluation_panel,
            text="Accuracy: --",
            style="Panel.TLabel",
            font=("Segoe UI", 13, "bold"),
        )
        self.accuracy_label.pack(anchor="w", pady=(0, 14))

        # Confusion matrix title
        matrix_title = ttk.Label(
            evaluation_panel,
            text="Confusion Matrix",
            style="Panel.TLabel",
            font=("Segoe UI", 10, "bold"),
        )
        matrix_title.pack(anchor="w", pady=(0, 6))

        # Confusion matrix table
        self.confusion_table = ttk.Treeview(
            evaluation_panel,
            columns=("predicted_ham", "predicted_spam"),
            show="headings tree",
            height=2,
        )
        self.confusion_table.heading("#0", text="Actual")
        self.confusion_table.heading("predicted_ham", text="Predicted Ham")
        self.confusion_table.heading("predicted_spam", text="Predicted Spam")
        self.confusion_table.column("#0", width=85, anchor="center")
        self.confusion_table.column("predicted_ham", anchor="center", width=120)
        self.confusion_table.column("predicted_spam", anchor="center", width=120)
        self.confusion_table.pack(fill="x")

    def build_chart_panel(self, parent):
        """Build the label distribution chart panel."""
        chart_panel = ttk.LabelFrame(
            parent,
            text="Label Distribution Chart",
            style="Panel.TLabelframe",
            padding=14,
        )
        chart_panel.pack(fill="both", expand=True, pady=(18, 0))

        self.chart_frame = ttk.Frame(chart_panel, style="Panel.TLabelframe")
        self.chart_frame.pack(fill="both", expand=True)

    def load_model(self):
        """Load and train the selected classifier model."""
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
                text=f"✓ Ready: {model_type} | Threshold: {spam_threshold:.2f}"
            )

            self.update_dataset_summary()
            self.update_evaluation_summary(evaluation)
            self.display_label_distribution_chart()
            self.clear_message()

        except (FileNotFoundError, ValueError) as error:
            self.status_label.config(text="❌ Model failed to load")
            messagebox.showerror("Error", str(error))

    def update_dataset_summary(self):
        """Update dataset statistics display."""
        dataset = self.classifier.dataset
        label_counts = dataset["label"].value_counts().to_dict()
        avg_length = dataset["message"].str.len().mean()

        self.total_records_label.config(text=f"Total records: {len(dataset)}")
        self.ham_count_label.config(text=f"✓ Ham messages: {label_counts.get('ham', 0)}")
        self.spam_count_label.config(text=f"⚠ Spam messages: {label_counts.get('spam', 0)}")
        self.avg_length_label.config(
            text=f"Average length: {avg_length:.0f} characters"
        )

    def update_evaluation_summary(self, evaluation):
        """Update the accuracy and confusion matrix display with evaluation results."""
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
                text=label,
                values=(row[0], row[1]),
            )

    def display_label_distribution_chart(self):
        """Generate and display the label distribution bar chart."""
        label_counts = (
            self.classifier.dataset["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
        )

        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # Create figure with better styling
        figure = Figure(figsize=(4.0, 2.6), dpi=100)
        figure.patch.set_facecolor("#2a2a3e")  # Match panel background
        
        axis = figure.add_subplot(111)
        axis.set_facecolor("#f0f0f0")

        labels = list(label_counts.index)
        values = list(label_counts.values)
        # Use better colors: blue for ham, red for spam
        colors = ["#51cf66", "#ff6b6b"]  # Green for ham, red for spam

        bars = axis.bar(labels, values, color=colors[:len(labels)], edgecolor="#333333", linewidth=1.5)

        # Styling
        axis.set_title("Message Label Distribution", fontsize=10, fontweight="bold", pad=10)
        axis.set_xlabel("Label", fontsize=9)
        axis.set_ylabel("Count", fontsize=9)
        axis.tick_params(axis="x", labelsize=9, rotation=0)
        axis.tick_params(axis="y", labelsize=9)
        axis.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height * 0.95,
                str(value),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

        figure.tight_layout()

        # Embed chart in tkinter
        self.chart_canvas = FigureCanvasTkAgg(figure, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def classify_message(self):
        """Classify the entered message and display the prediction."""
        if self.classifier is None:
            messagebox.showerror("Error", "The model is not ready. Please wait for loading.")
            return

        message = self.message_input.get("1.0", tk.END).strip()

        if not message:
            messagebox.showwarning("Missing message", "Please enter an email message to classify.")
            return

        try:
            label, confidence = self.classifier.predict(message)

            if label == "spam":
                result_text = "🚨 SPAM"
                result_color = "#ff6b6b"
            else:
                result_text = "✓ HAM"
                result_color = "#51cf66"

            self.result_label.config(
                text=result_text,
                foreground=result_color,
            )
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1%}",
            )

        except ValueError as error:
            messagebox.showerror("Prediction Error", str(error))

    def clear_message(self):
        """Clear the message input and reset prediction display."""
        self.message_input.delete("1.0", tk.END)
        self.result_label.config(
            text="Awaiting classification...",
            foreground="#ffffff",
        )
        self.confidence_label.config(text="Confidence: --")

    def change_model(self, event=None):
        """Load a different classifier model when user changes selection."""
        self.load_model()


def main():
    root = tk.Tk()
    EmailClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()