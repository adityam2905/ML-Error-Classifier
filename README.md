# ML-Based Code Error Classifier

A Streamlit app that takes C++ or Python code, collects the errors it produces, and uses a text classifier to label each error as **lexical**, **syntactic** or **semantic**.

- **C++:** the code is compiled with `g++` (never run). Each `error:` line from the compiler is classified.
- **Python:** the code is run in a subprocess. The final line of the traceback (e.g. `NameError: name 'x' is not defined`) is classified.

For each error the app shows the predicted category, the model's confidence, and the probability for every category.

## Results

The C++ classifier is the main result. The Python classifier is included, but is no better than a simple rule (see below).

| Language | Model | Distinct messages | 5-fold CV accuracy | Held-out test accuracy |
|---|---|---|---|---|
| **C++** | Multinomial Naive Bayes | 666 | **86%** | 95% (134 messages) |
| Python | Random Forest | 79 | 65% | 62% (16 messages) |

How these numbers are measured:

- Duplicates are removed *after* preprocessing and *before* the 80/20 train/test split. No test message also appears in the training data, so the scores reflect messages the model has not seen.
- The best of four models (Logistic Regression, Random Forest, Multinomial NB, linear SVM) is chosen by 5-fold cross-validation on the training split, then scored once on the test split.
- The C++ test score comes from a single split, so the cross-validation figure is the more reliable estimate.

**Why Python is weak:** the 750 rows in `python_error_dataset.csv` differ mostly by a `[line:… col:… id:…]` suffix. Once that is removed, only 79 distinct messages remain. A rule that predicts from the exception name alone (`NameError`, `TypeError`, …) scores 62% on the same split, the same as the model. The training script prints this baseline so the comparison is always visible.

## Limitations

- **C++ runtime errors are out of scope.** The app compiles C++ code but never runs it, so only compile-time errors reach the classifier. The 119 runtime-only rows in `cpp_error_dataset.csv` (`runtime error: …`, segmentation faults) are dropped when training.
- **The labels are this project's own scheme, not compiler phases.** For example, "undeclared identifier" is labelled *lexical*, but a compiler detects it during semantic analysis (name lookup). Read the labels as rough error categories, not as the compiler stage that reports each error.
- **Python code is executed.** Submitted Python code runs on the machine hosting the app, with a 5-second timeout and no sandbox. Only run the app locally, with code you trust.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `analyzers.py` | Compiles C++ / runs Python and extracts the error messages |
| `error_classifier.py` | Shared preprocessing, training, evaluation and prediction |
| `mlbec3.py` | Training script: deduplicates, splits, trains, evaluates and saves the models |
| `cpp_error_dataset.csv` | C++ training data (`error_message`, `error_type`) |
| `python_error_dataset.csv` | Python training data (`error_message`, `error_type`) |

## Setup

Requires Python 3.10+ and, for C++ analysis, `g++` on your `PATH`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

NLTK data (tokenizer, stopwords, WordNet) downloads automatically the first time the classifier is imported.

## Train the models

```bash
python mlbec3.py
```

Choose `1` (C++), `2` (Python) or `3` (both). The script prints the cross-validation scores, a classification report, and (for Python) the exception-name baseline. It writes:

- `cpp_error_classifier.pkl`, `python_error_classifier.pkl`: the trained models the app loads
- `cpp_confusion_matrix.png`, `python_confusion_matrix.png`: test-set confusion matrices

These files are generated and are not committed.

## Run the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, pick a language in the sidebar, upload a file or paste code, and click **Analyze Code**.

## How it works

1. **Extract errors.** For C++, a regex pulls the message text from each `file:line:col: error:` line, discarding `note:` and context lines. For Python, the last non-traceback line of stderr is used.
2. **Preprocess.** File paths, line/column numbers, hex addresses and compiler keywords are removed. Quoted identifiers become a `_TOKEN_` placeholder, so `'x' was not declared` and `'count' was not declared` look the same. The text is then tokenized, stopwords are removed, and words are lemmatized with NLTK.
3. **Vectorize and classify.** A TF-IDF vectorizer (unigrams and bigrams) feeds the selected model. The same preprocessing runs during training and in the app.
