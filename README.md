# COMP 472 Mini Project 2: Email classifier
This project is an AI-powered email filtering system for COMP 472. The program will convert email text into numerical features, train a machine learning model with a a CSV SMS Spam Collection dataset, predict whether new emails are spam or not, display confidence levels and evaluate the model performance.
## Features
- Loads a data/TBD.csv file with label,message columns
- Uses pandas for CSV loading
- Uses TfidfVectorizer from scikit-learn to convert text into numerical features
- Trains a machine learning classifier using Logistic Regression or Naive Bayes from scikit-learn
- Evaluates performance of the model by displaying its accuracy and a confusion matrix
- Displays the predicted label and confidence score
- Generates a chart showing the number of spam and non-spam messages using matplotlib
- Maintains the prediction recursively by continuously accepting user input until the user quit the program
## Project Structure
email_classifier/
├── data/
│   └── TBD.csv                     # SMS/Email dataset (label, message)
├── src/
│   ├── __init__.py
│   ├── main.py                     # CLI entry point
│   ├── classifier.py               # ML classifier logic
│   ├── feature_extraction.py       # TfidfVectorizer
│   ├── evaluation.py               # Accuracy, confusion matrix
│   └── visualization.py            # Matplotlib charts
├── gui/
│   ├── __init__.py
│   └── gui_app.py                  # Tkinter GUI (optional)
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py
│   └── test_feature_extraction.py
├── requirements.txt
├── README.md
└── .gitignore

## Setup in VS Code

### Windows 


### macOS/ Linux

## Architecture

## How it works 

## Optional GUI

## Example Run

## Test Input

## Running the basic test


