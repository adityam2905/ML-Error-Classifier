# mlbec3.py
# Unified ML Training Script for C++ and Python Error Classification
# (Now imports the classifier logic from error_classifier.py)

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

# Import the new classifier classes
from error_classifier import CppErrorClassifier, PythonErrorClassifier

def main_training(language: str):
    """
    Main function to load data, train, evaluate, and save a model.
    """
    if language == 'cpp':
        dataset_path = "cpp_error_dataset.csv"
        model_save_path = "cpp_error_classifier.pkl"
        trainer = CppErrorClassifier()
        print("\n--- Initializing C++ Error Classifier Training ---")
    elif language == 'python':
        dataset_path = "python_error_dataset.csv"
        model_save_path = "python_error_classifier.pkl"
        trainer = PythonErrorClassifier()
        print("\n--- Initializing Python Error Classifier Training ---")
    else:
        print(f"Unknown language: {language}")
        return

    try:
        # 1. Load Dataset
        print(f"Loading dataset from {dataset_path}...")
        X, y = trainer.load_dataset(dataset_path)
        print(f"Dataset loaded: {len(X)} samples.")
        print("Class distribution:\n", y.value_counts(normalize=True))

        # 2. Split Data
        print("Splitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Fit Preprocessors (Vectorizer, Encoder)
        trainer.fit_vectorizer_and_encoder(X_train, y_train)

        # 4. Train Model
        trainer.train(X_train, y_train)

        # 5. Evaluate Model
        trainer.evaluate(X_test, y_test)

        # 6. Save Model
        trainer.save_model(model_save_path)

        print(f"\n--- {language.upper()} training complete. Model saved to {model_save_path} ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the dataset file exists.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    cpp_dataset = "cpp_error_dataset.csv"
    py_dataset = "python_error_dataset.csv"

    # Check which datasets exist
    has_cpp = os.path.exists(cpp_dataset)
    has_py = os.path.exists(py_dataset)

    if not has_cpp and not has_py:
        print(f"Error: Neither '{cpp_dataset}' nor '{py_dataset}' found.")
        print("Please add your datasets to the directory to run training.")
        sys.exit(1)

    options = []
    if has_cpp:
        options.append("(1) C++")
    if has_py:
        options.append("(2) Python")
    if has_cpp and has_py:
        options.append("(3) Both")

    print("--- Error Classifier Training ---")
    choice = input(f"Train for {'  '.join(options)} : ").strip()

    if choice in ['1', '3'] and has_cpp:
        main_training('cpp')

    if choice in ['2', '3'] and has_py:
        main_training('python')

    if choice not in ['1', '2', '3']:
        print("Invalid choice. Exiting.")