# error_classifier.py
"""
Core machine learning logic for both C++ and Python error classification.

Keeping the classifiers here means:
1.  The training script (mlbec3.py) and the app (analyzers.py) share them.
2.  The exact same text preprocessing runs during training and live analysis.
3.  Each language gets its own text cleaning (e.g., removing file paths,
    line numbers, and quoted identifiers).
"""

import pandas as pd
import numpy as np
import re
import nltk
import os
import pickle
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

# ----------------------------------------------------
# NLTK Data Download
# (Called on import, ensuring data is available for
#  both training and analysis)
# ----------------------------------------------------


def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    print("Checking NLTK data...")
    data_downloaded = False
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading missing NLTK resource: {name}")
            nltk.download(name)
            data_downloaded = True
    if not data_downloaded:
        print("All NLTK data is up-to-date.")


# Call this function once when the module is imported
download_nltk_data()

# ----------------------------------------------------
# Base Classifier (Handles all shared ML logic)
# ----------------------------------------------------


class BaseErrorClassifier:
    """
    Base class for error classification. Handles data loading,
    model training, evaluation, and prediction.
    """

    def __init__(self):
        # Use n-grams (1, 2) to capture word pairs like "name error"
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words='english')
        self.label_encoder = LabelEncoder()
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'MultinomialNB': MultinomialNB(alpha=0.1),
            'SVC': SVC(probability=True, C=1.0, kernel='linear', random_state=42)
        }
        self.best_model = None
        self.best_model_name = "None"
        self.is_fitted = False
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def load_dataset(self, filepath: str):
        """Loads a dataset from a CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        df = pd.read_csv(filepath)
        df = df.dropna(subset=['error_message', 'error_type'])
        df['error_message'] = df['error_message'].astype(str)
        df['error_type'] = df['error_type'].astype(str)
        return df['error_message'], df['error_type']

    def _base_preprocess(self, text: str) -> str:
        """
        Core text preprocessing: tokenization, stopword removal, lemmatization.
        This is called *after* language-specific cleaning.
        """
        # 1. Tokenize
        tokens = word_tokenize(text.lower())
        
        # 2. Lemmatize and remove stopwords/punctuation
        processed_tokens = []
        for token in tokens:
            # Keep the special _TOKEN_
            if token == '_token_':
                 processed_tokens.append(token)
            elif token.isalpha() and token not in self.stop_words:
                processed_tokens.append(self.lemmatizer.lemmatize(token))
        
        return " ".join(processed_tokens)

    def preprocess_text(self, text: str) -> str:
        """
        Language-specific preprocessing. This MUST be overridden by subclasses.
        This is the main entry point for preprocessing.
        """
        raise NotImplementedError(
            "Subclass must implement 'preprocess_text'")

    def fit_vectorizer_and_encoder(self, X_train: pd.Series, y_train: pd.Series):
        """
        Fits the TF-IDF vectorizer and LabelEncoder on the training data.
        Applies the full preprocessing pipeline.
        """
        print("Fitting Vectorizer and Label Encoder...")
        # Apply the full preprocessing pipeline (specific + base)
        X_train_processed = X_train.apply(self.preprocess_text)
        
        # Fit vectorizer
        self.vectorizer.fit(X_train_processed)
        print(f"Vectorizer fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Fit label encoder
        self.label_encoder.fit(y_train)
        print(f"Label Encoder fitted. Classes: {list(self.label_encoder.classes_)}")

    def transform_text(self, X_data: pd.Series):
        """Applies preprocessing and TF-IDF transformation."""
        X_processed = X_data.apply(self.preprocess_text)
        return self.vectorizer.transform(X_processed)

    def train(self, X_train: pd.Series, y_train: pd.Series):
        """
        Trains all models and selects the best one using cross-validation.
        """
        if self.vectorizer is None or self.label_encoder is None:
            raise Exception(
                "Vectorizer and LabelEncoder must be fitted first. Call 'fit_vectorizer_and_encoder'.")

        X_train_vec = self.transform_text(X_train)
        y_train_enc = self.label_encoder.transform(y_train)
        
        print("\n--- Starting Model Training ---")
        best_score = 0.0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.models.items():
            try:
                print(f"Training {name}...")
                scores = cross_val_score(
                    model, X_train_vec, y_train_enc, cv=skf, scoring='accuracy')
                mean_score = np.mean(scores)
                print(
                    f"  {name} Cross-Val Accuracy: {mean_score:.4f} (+/- {np.std(scores):.4f})")

                if mean_score > best_score:
                    best_score = mean_score
                    self.best_model_name = name
            except Exception as e:
                print(f"  Failed to train {name}: {e}")

        # Final fit of the best model on all training data
        self.best_model = self.models[self.best_model_name]
        self.best_model.fit(X_train_vec, y_train_enc)
        self.is_fitted = True
        print(
            f"\n--- Best Model Selected: {self.best_model_name} (Accuracy: {best_score:.4f}) ---")

    def evaluate(self, X_test: pd.Series, y_test: pd.Series):
        """Evaluates the best model on the test set."""
        if not self.best_model:
            print("No model trained. Please call 'train' first.")
            return

        print("\n--- Model Evaluation on Test Set ---")
        X_test_vec = self.transform_text(X_test)
        y_test_enc = self.label_encoder.transform(y_test)
        
        y_pred_enc = self.best_model.predict(X_test_vec)
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)

        # Classification Report
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_test, y_pred,
                                  labels=self.label_encoder.classes_)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_)
            plt.title(f'Confusion Matrix for {self.best_model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            # Use a language-specific name for the confusion matrix
            if isinstance(self, CppErrorClassifier):
                cm_filename = 'cpp_confusion_matrix.png'
            elif isinstance(self, PythonErrorClassifier):
                cm_filename = 'python_confusion_matrix.png'
            else:
                cm_filename = 'confusion_matrix.png'
            plt.savefig(cm_filename)
            print(f"Confusion matrix saved to '{cm_filename}'")
        except Exception as e:
            print(f"Could not generate confusion matrix: {e}")

    def save_model(self, filepath: str):
        """Saves the entire fitted classifier instance to a pickle file."""
        if not self.is_fitted:
            print("Warning: Saving a model that has not been trained.")
        print(f"Saving model to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print("Model saved successfully.")

    def _predict(self, vectorized_text):
        """Internal prediction helper."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        pred_enc = self.best_model.predict(vectorized_text)
        return self.label_encoder.inverse_transform(pred_enc)

    def _predict_proba(self, vectorized_text):
        """Internal probability helper."""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        probabilities = self.best_model.predict_proba(vectorized_text)
        # Map probabilities to class names
        class_probs = []
        for prob_array in probabilities:
            class_probs.append(
                dict(zip(self.label_encoder.classes_, prob_array))
            )
        return class_probs

    def predict_single(self, raw_error_text: str) -> str:
        """Predicts the class for a single raw error message string."""
        vectorized_text = self.transform_text(pd.Series([raw_error_text]))
        prediction = self._predict(vectorized_text)
        return prediction[0]

    def predict_proba_single(self, raw_error_text: str) -> dict:
        """Predicts probabilities for a single raw error message string."""
        vectorized_text = self.transform_text(pd.Series([raw_error_text]))
        probabilities = self._predict_proba(vectorized_text)
        return probabilities[0]


# ----------------------------------------------------
# C++ Specific Classifier
# ----------------------------------------------------

class CppErrorClassifier(BaseErrorClassifier):
    """
    C++ Error Classifier.
    Implements C++-specific text preprocessing to clean compiler output.
    """

    # The app only compiles C++ (it never runs the binary), so runtime
    # diagnostics can never reach the classifier. Drop them from training.
    RUNTIME_ONLY = re.compile(
        r'^\s*(runtime (error|warning)|segmentation fault)'
        r'|fatal error: (segmentation fault|undefined behavior detected at runtime)',
        re.I)

    def load_dataset(self, filepath: str):
        X, y = super().load_dataset(filepath)
        keep = ~X.str.contains(self.RUNTIME_ONLY)
        print(f"Dropped {(~keep).sum()} runtime-only rows (C++ code is compiled, never run).")
        return X[keep], y[keep]

    def preprocess_text(self, text: str) -> str:
        """
        Applies C++-specific regex cleaning *before* base preprocessing.
        """
        # 1. Remove file paths, line/column numbers
        cleaned = re.sub(
            r'(?:[a-zA-Z]:\\[^:\n\s]+|(?:\.\/|\/)[^:\n\s]+):\d+:\d+:', ' ', text, flags=re.I)
        
        # 2. Remove compiler noise keywords
        cleaned = re.sub(r'\b(error|warning|note):\s*', ' ', cleaned, flags=re.I)
        
        # 3. Replace quoted entities with a placeholder token
        cleaned = re.sub(r"[\"\'\`“‘](.*?)[\"\'\`”’]", ' _TOKEN_ ', cleaned)
        
        # 4. Remove template instantiation/requirement lines
        cleaned = re.sub(r'\b(instantiated|required)\s+from\s.*', ' ', cleaned, flags=re.I)
        
        # 5. Remove function/context lines
        cleaned = re.sub(r'\b(in|at)\s+(function|member function|constructor|destructor)\s.*', ' ', cleaned, flags=re.I)
        
        # 6. Remove "in file..." lines
        cleaned = re.sub(r'\b(in|from)\s+file\s.*', ' ', cleaned, flags=re.I)

        # 7. Remove "candidate..." lines
        cleaned = re.sub(r'\bcandidate(s?):\s.*', ' ', cleaned, flags=re.I)

        # 8. Remove "with..." blocks
        cleaned = re.sub(r'\bwith\s+\[.*\]', ' ', cleaned, flags=re.I)

        # 9. Remove hexadecimal addresses
        cleaned = re.sub(r'0x[a-fA-F0-9]+', ' ', cleaned)
        
        # 10. Remove C++ carets and tildes from diagnostics
        cleaned = re.sub(r'[\^\~]+', ' ', cleaned)

        # 11. Remove ellipses
        cleaned = re.sub(r'\.{3,}', ' ', cleaned)
        
        # 12. Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # 13. Call the base preprocessor (tokenize, lemmatize, stopwords)
        return self._base_preprocess(cleaned)


# ----------------------------------------------------
# Python Specific Classifier
# ----------------------------------------------------

class PythonErrorClassifier(BaseErrorClassifier):
    """
    Python Error Classifier.
    Implements Python-specific text preprocessing to clean interpreter output.
    """

    def preprocess_text(self, text: str) -> str:
        """
        Applies Python-specific regex cleaning *before* base preprocessing.
        This version KEEPS the error name (e.g., "NameError") as a feature.
        """
        
        # 1. Keep the error name as a feature
        #    "NameError: 'x' not defined" -> "NameError 'x' not defined"
        #    We replace the colon with a space.
        cleaned = re.sub(r'^([a-zA-Z_]*Error):', r'\1 ', text, flags=re.I)
        
        # 2. Handle SyntaxError differently, as it's often multi-line in trace
        #    If the input text is just "SyntaxError: invalid syntax"
        if "SyntaxError:" in cleaned:
             # Make it a single token "syntaxerror"
             cleaned = cleaned.replace("SyntaxError:", "syntaxerror ")

        # 3. Replace quoted tokens
        #    "NameError 'x' not defined" -> "NameError _TOKEN_ not defined"
        cleaned = re.sub(r"[\"\'\`“‘](.*?)[\"\'\`”’]", ' _TOKEN_ ', cleaned)

        # 4. Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 5. Call the base preprocessor (which will lowercase, lemmatize, etc.)
        #    "NameError _TOKEN_ not defined" -> "nameerror _token_ defined"
        return self._base_preprocess(cleaned)