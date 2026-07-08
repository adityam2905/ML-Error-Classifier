# ML-Based Code Error Classification

A Streamlit application that detects and classifies C++ and Python code errors using machine learning.

## Features

- Analyze **C++** and **Python** code from uploaded files or pasted text.
- Classify detected errors into categories (for example: lexical, syntactic, semantic).
- Show prediction confidence and probability distribution.
- Train models from provided datasets.

## Project Structure

- `app.py` - Streamlit web app
- `analyzers.py` - C++/Python runtime analyzers
- `error_classifier.py` - shared ML classifier logic
- `mlbec3.py` - model training script
- `cpp_error_dataset.csv` - C++ training dataset
- `python_error_dataset.csv` - Python training dataset

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Models

Before running the app, train model files:

```bash
python mlbec3.py
```

This generates:
- `cpp_error_classifier.pkl`
- `python_error_classifier.pkl`

## Run the App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, select language, then upload/paste code and click **Analyze Code**.

## Notes

- C++ analysis requires `g++` to be installed and available in `PATH`.
- NLTK resources are auto-downloaded on first run of the classifier module.
