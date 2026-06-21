# COMP 472 Mini Project 2: Email Classifier

An AI-powered email spam classification system using machine learning and natural language processing. This project demonstrates a complete ML pipeline: data loading, feature extraction, model training, evaluation, and interactive predictions.

## ✨ Features

- **Dataset Management**: Loads SMS Spam Collection dataset from CSV with automatic validation and cleaning
- **Feature Extraction**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization from scikit-learn
- **Classification Models**: 
  - Logistic Regression with regularization
  - Multinomial Naive Bayes
- **Performance Evaluation**: Displays accuracy metrics and confusion matrix
- **Prediction Interface**: 
  - Command-line: Interactive loop accepting user input until exit
  - GUI (Tkinter): Modern dark-themed interface with real-time predictions
- **Data Visualization**: Bar chart showing spam/ham distribution
- **Comprehensive Testing**: 28 unit tests covering all major components

## Project Structure

```
email_classifier/
├── data/
│   └── spam.csv                 # SMS Spam Collection dataset
├── src/
│   ├── main.py                  # CLI entry point
│   ├── classifier.py            # Email classifier coordinator
│   ├── dataset.py               # Data loading & validation
│   ├── vectorizer.py            # TF-IDF feature extraction
│   ├── training.py              # Model creation & training
│   ├── evaluation.py            # Accuracy & confusion matrix
│   └── visualization.py         # Charts and plots
├── gui/
│   └── app.py                   # Tkinter GUI application
├── tests/
│   ├── test_classifier.py
│   ├── test_dataset.py
│   ├── test_vectorizer.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_visualization.py
├── requirements.txt
└── README.md
```

## Quick Start

### Installation (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Installation (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-Line Interface

```bash
python -m src.main
```

Follow the prompts to:
1. Choose a classifier model (Logistic Regression or Naive Bayes)
2. View dataset statistics
3. Review model evaluation metrics
4. Enter email messages for classification

### Graphical User Interface

```bash
python -m gui.app
```

Features:
- Model selection dropdown (switch between models in real-time)
- Paste or type email text for instant classification
- View prediction with confidence score
- Dataset statistics panel
- Model evaluation (accuracy + confusion matrix)
- Label distribution chart

## Architecture

The project uses a modular, object-oriented design that separates concerns:

- **EmailClassifier**: Main coordinator connecting all pipeline stages
- **SpamDataset**: Handles CSV loading, validation, and cleaning
- **TextVectorizer**: Wraps scikit-learn's TfidfVectorizer for feature extraction
- **Training utilities**: Model creation, data splitting, and fitting
- **Evaluation utilities**: Accuracy calculation and confusion matrix
- **Visualization**: Matplotlib charts for data exploration

This modular design makes it easy to:
- Swap between different models without changing the interface
- Test each component independently
- Reuse components in both CLI and GUI applications

## Machine Learning Pipeline

### 1. Data Loading & Validation
- Loads CSV file using pandas
- Validates that `label` and `message` columns exist
- Removes rows with missing values
- Normalizes labels to lowercase
- Handles multiple character encodings (UTF-8, Latin-1, ISO-8859-1)

### 2. Feature Extraction (TF-IDF)
- Converts raw text messages into numerical features
- TF-IDF gives higher weight to distinctive words and lower weight to common words
- Removes stop words (common English words like "the", "and", "is")
- Output: Sparse matrix of feature vectors

### 3. Train/Test Split
- Splits dataset into 80% training and 20% test sets
- Uses **stratified splitting** to maintain spam/ham ratio in both sets
- Critical for preventing data leakage and proper evaluation

### 4. Model Training
Two classifier options available:

**Logistic Regression**
- Linear probabilistic model
- Fast training and prediction
- Interpretable decision boundaries
- Threshold: 0.35 (more aggressive spam detection)

**Naive Bayes**
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Works well with text data
- Threshold: 0.20 (very aggressive spam detection)

### 5. Prediction & Confidence
- Model returns probability estimates for each class
- User-configurable threshold determines final classification
- Confidence score = probability of predicted class
- Example: If spam probability is 0.92 and threshold is 0.40, prediction is "SPAM" with 92% confidence

### 6. Evaluation Metrics
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: 
  - True Negatives (ham correctly identified)
  - False Positives (ham misclassified as spam)
  - False Negatives (spam misclassified as ham)
  - True Positives (spam correctly identified)

## GUI Features

The Tkinter GUI provides an intuitive interface for spam classification:

### Left Panel
- **Email Input**: Text area for pasting/typing email messages
- **Classify Button**: Predicts the label instantly
- **Clear Button**: Resets the input and results
- **Result Display**: Shows prediction with color coding (🚨 SPAM in red, ✓ HAM in green)

### Right Panel
- **Dataset Statistics**: Total messages, ham/spam counts, average message length
- **Model Evaluation**: Accuracy percentage and confusion matrix
- **Distribution Chart**: Bar chart showing dataset composition

### Model Selection
- Dropdown menu to switch between Logistic Regression and Naive Bayes
- Automatically retrains the model and updates all displays
- Status indicator shows current model and decision threshold

## Testing

Run the test suite:

```bash
python -m pytest -v           # Verbose output
python -m pytest -q           # Quick output
python -m pytest tests/       # Run specific test file
```

Tests cover:
- Dataset loading and validation
- Data normalization and cleaning
- Model training and prediction
- TF-IDF vectorization
- Evaluation metrics calculation
- Visualization generation

## Code Quality

This project includes:
- **Comprehensive docstrings** on all classes and functions (Google-style)
- **Type hints** for function parameters and return values
- **Error handling** for common failures (missing files, invalid input, empty data)
- **Modular design** separating concerns into focused components
- **Input validation** at every boundary
- **Clean comments** explaining complex logic
- **Consistent naming** following Python conventions (snake_case for functions/variables, PascalCase for classes)

## Example Run

```text
==================================================
  Email Classifier - COMP 472
==================================================

Choose a model:
  1. Logistic Regression
  2. Naive Bayes
Model [1]: 1

--- Dataset Information ---
Total records: 5572
Label distribution: {'ham': 4825, 'spam': 747}
Average message length: 80 characters
Model: logistic
Spam threshold: 0.35

Training Model logistic...

--- Evaluation ---
Accuracy: 97.85%
Confusion matrix:
  Classes: ['ham', 'spam']
  ham: [965, 1]
  spam: [23, 126]
------------------------

Enter an email message to classify, or type 'quit' to exit.
Message: Congratulations! You won $5000.
Prediction: SPAM | Confidence: 45.38%

Message: Please submit your project by tomorrow.
Prediction: HAM | Confidence: 94.71%

Message: quit
Goodbye!
```

## Test Input
Try these:
```text
Win a free vacation now!
Your appointment is scheduled for tomorrow.
Congratulations! You have won a free iPhone.
Please submit your assignment before Friday.
Claim your prize now!
Meeting moved to 2 PM tomorrow.
Win a FREE iPhone today!
Reservation at 6pm next Saturday
Congratulations! Claim your tickets now!
```

## Running the basic test
```bash
python -m pytest
```
The included test checks every modules and features of the program. They uses predifined valid CSV created on the spot so they don't load the whole spam.csv dataset every time.

## Screenshots

### CLI
![CLI training and chart](docs/screenshots/cli-diagram.png)
![CLI classification results](docs/screenshots/cli-result.png)

### GUI

#### Example 1: Spam - Logistic Model
![sc1 spam classification](docs/screenshots/spam-logistic.png)

#### Example 2: Spam - Naive Bayes Model
![sc2 spam classification](docs/screenshots/spam-bayes.png)

#### Example 3: Ham - Logistic Model
![sc3 ham classification](docs/screenshots/ham-logistic.png)

#### Example 4: Ham - Naive Bayes Model
![sc4 ham classification](docs/screenshots/ham-bayes.png)
