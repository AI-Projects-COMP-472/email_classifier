# COMP 472 Mini Project 2: Email Classifier

This project implements a command-line email filtering system for COMP 472. It trains a machine learning classifier on a spam dataset, evaluates its performance, and predicts whether new messages are spam or ham.

## Features

- Loads `data/spam.csv` with `label` and `message` columns
- Uses `pandas` for dataset loading and validation
- Converts email text to TF-IDF features using `scikit-learn`
- Supports Logistic Regression and Multinomial Naive Bayes
- Evaluates model accuracy, confusion matrix, and classification report
- Predicts spam/ham labels with confidence scores
- Generates a bar chart showing spam vs non-spam counts with `matplotlib`
- Accepts continuous user input until the user exits

## Project Structure

```text
email_classifier/
├── data/
│   └── spam.csv
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── classifier.py
│   ├── conversion.py
│   ├── evaluation.py
│   └── visualization.py
├── gui/
│   ├── __init__.py
│   └── app.py
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py
│   ├── test_conversion.py
│   └── test_visualization.py
├── requirements.txt
├── README.md
└── short_reflection.txt
```

## Requirements

- Python 3.8 or later
- `pandas`
- `scikit-learn`
- `matplotlib`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Create and activate a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the main program from the project root:

```bash
python -m src.main
```

The program will:

1. Prompt for a model choice
2. Train the selected classifier
3. Display dataset information and evaluation metrics
4. Save a spam/ham distribution chart
5. Allow repeated input for new email classification

## Demo Script

A demo script is available for a quick sample run:

```bash
python demo.py
```

This will train the model, print evaluation results, save `spam_distribution_demo.png`, and classify example messages.

## Testing

Run unit tests with:

```bash
python -m pytest -q
```

## Example Run

After starting the program, type an email message and press Enter.
Then enter `quit` to exit.

## Notes

- `src/main.py` is the main CLI entry point.
- `src/classifier.py` manages dataset loading, training, prediction, and evaluation.
- `src/visualization.py` generates the required spam/ham distribution chart.
