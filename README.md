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
```email_classifier/
├── data/
│   └── TBD.csv
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── classifier.py
│   ├── conversion.py
│   ├── evaluation.py
│   └── visualization.py
├── gui/
│   ├── __init__.py
│   └── gui_app.py
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py
│   └── test_conversion.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup in VS Code

### Windows 


### macOS/ Linux

## Architecture

## How it works 

## Optional GUI

## Example Run

## Test Input

## Running the basic test


