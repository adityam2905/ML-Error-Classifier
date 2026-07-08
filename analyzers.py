# analyzers.py
"""
Contains all code analyzers for the Streamlit app.
- CppCodeErrorAnalyzer: Compiles C++ code and classifies errors.
- PythonCodeErrorAnalyzer: Executes Python code and classifies errors.

These classes have been updated to:
1.  Import the classifier definitions from error_classifier.py
    (This is necessary for pickle.load() to work).
2.  Use the new, improved error parsing for C++ and Python.
3.  Call the classifier's built-in predict methods, which now handle
    all text preprocessing automatically.
"""
import subprocess
import os
import tempfile
import shutil
import pickle
import sys
import re
from typing import List, Dict, Tuple, Optional, Any

# IMPORTANT: This import is necessary for pickle to deserialize
# the saved model objects. Even if not "used" directly,
# it defines the classes that pickle needs to find.
try:
    from error_classifier import CppErrorClassifier, PythonErrorClassifier
except ImportError:
    print("ERROR: Could not import from error_classifier.py")
    print("Please ensure error_classifier.py is in the same directory.")
    # Define dummy classes to allow the rest of the file to be parsed,
    # but it will fail at runtime.
    class CppErrorClassifier: pass
    class PythonErrorClassifier: pass

# --- Base Analyzer Class ---

class BaseCodeErrorAnalyzer:
    """Base class for code analyzers."""
    
    def __init__(self, model_path: str):
        self.classifier: Optional[Any] = None # Will be Cpp/PythonErrorClassifier
        self.model_path = model_path
        self.load_classifier()

    def load_classifier(self):
        """Load the trained error classification model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file '{self.model_path}' not found. "
                    f"Please run 'python mlbec3.py' to train and create it."
                )

            with open(self.model_path, 'rb') as f:
                # pickle.load() will work because we imported the class definitions
                self.classifier = pickle.load(f)
            
            if not hasattr(self.classifier, 'predict_single') or \
               not hasattr(self.classifier, 'predict_proba_single'):
                raise TypeError(
                    f"The loaded model at '{self.model_path}' is not a valid "
                    "classifier. It's missing 'predict_single' or 'predict_proba_single'. "
                    "Please retrain with the new 'mlbec3.py'."
                )
            
            print(f"Successfully loaded model '{self.model_path}' "
                  f"({self.classifier.best_model_name})")

        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            self.classifier = None
        except (pickle.UnpicklingError, EOFError, TypeError, AttributeError) as e:
            print(f"Error deserializing model from '{self.model_path}': {e}")
            print("This may be an old model file. Please retrain with 'mlbec3.py'.")
            self.classifier = None
        except Exception as e:
            print(f"An unexpected error occurred loading model: {e}")
            self.classifier = None

    def analyze(self, code: str) -> Dict:
        """Main analysis method. To be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement 'analyze'")

    def _classify_errors(self, error_messages: List[str]) -> List[Dict]:
        """
        Classifies a list of error messages using the loaded model.
        """
        if self.classifier is None:
            print("Classifier not loaded. Returning 'unknown' for all errors.")
            return [{
                'error_number': i + 1,
                'error_message': error_msg,
                'predicted_type': 'unknown (model not loaded)',
                'confidence': 0.0,
                'all_probabilities': {}
            } for i, error_msg in enumerate(error_messages)]

        classifications = []
        for i, error_msg in enumerate(error_messages):
            if not error_msg.strip():
                continue
            
            try:
                # The classifier object handles ALL preprocessing
                predicted_type = self.classifier.predict_single(error_msg)
                confidence_dict = self.classifier.predict_proba_single(error_msg)
                confidence = confidence_dict.get(predicted_type, 0.0)

                classifications.append({
                    'error_number': i + 1,
                    'error_message': error_msg,
                    'predicted_type': predicted_type,
                    'confidence': confidence,
                    'all_probabilities': confidence_dict
                })
            except Exception as e:
                classifications.append({
                    'error_number': i + 1,
                    'error_message': error_msg,
                    'predicted_type': 'unknown (prediction failed)',
                    'confidence': 0.0,
                    'all_probabilities': {},
                    'error': str(e)
                })

        return classifications

    def _generate_summary(self, classifications: List[Dict]) -> Dict:
        """Generate a summary of error types found."""
        summary = {}
        if self.classifier:
            # Get all possible labels from the encoder
            all_labels = self.classifier.label_encoder.classes_
            summary = {label.lower(): 0 for label in all_labels}
        
        summary['unknown'] = 0 # Always add unknown

        for cls in classifications:
            pred_type = cls['predicted_type'].lower()
            if pred_type in summary:
                summary[pred_type] += 1
            elif 'unknown' in pred_type:
                 summary['unknown'] += 1
            # Handle cases where pred_type might be new/unexpected
            elif pred_type not in summary:
                 summary[pred_type] = 1

        # Remove zero-count labels, except for 'unknown' which we always show
        summary_filtered = {k: v for k, v in summary.items() if v > 0 or k == 'unknown'}
        if 'unknown' not in summary_filtered:
            summary_filtered['unknown'] = 0

        return summary_filtered


# --- C++ Code Analyzer ---

class CppCodeErrorAnalyzer(BaseCodeErrorAnalyzer):
    
    def __init__(self, model_path='cpp_error_classifier.pkl'):
        """Initialize the analyzer with a trained model."""
        super().__init__(model_path)
        self.compiler_available = self._check_compiler()

    def _check_compiler(self, compiler='g++') -> bool:
        """Check if g++ compiler is available."""
        return shutil.which(compiler) is not None

    def _parse_output(self, output: str) -> List[str]:
        """
        *** NEW: Parse g++ output to find *only* the 'error:' lines. ***
        This is a much cleaner approach, discarding 'note:' and context lines,
        which are likely confusing the model (Data Mismatch).
        """
        if not output:
            return []
        
        error_lines = []
        # Regex to find lines that are C++ errors
        # (?:...|...): non-capturing group for different path types
        # [^:\n\s]+: matches file paths
        # :\d+:\d+: matches line:column
        # \s*error:\s*: matches 'error:' with flexible spacing
        # (.*): captures the actual error message
        error_regex = re.compile(
            r'(?:[a-zA-Z]:\\[^:\n\s]+|(?:\.\/|\/|)[^:\n\s]+):\d+:\d+:\s*error:\s*(.*)', 
            re.I
        )
        
        for line in output.splitlines():
            match = error_regex.search(line)
            if match:
                # Add the captured error message (group 1)
                error_message = match.group(1).strip()
                error_lines.append(error_message)
        
        # If no explicit `error:` lines were found (e.g., linker error)
        # fall back to a simpler split.
        if not error_lines and output.strip():
             # Check for common linker error "undefined reference to"
            if "undefined reference to" in output:
                # This is a good, clean signal for linker errors
                return ["linker error undefined reference"]
            # Fallback: return the first few non-empty lines
            fallback_lines = [line.strip() for line in output.splitlines() if line.strip()][:3]
            return fallback_lines

        return error_lines

    def analyze(self, code: str) -> Dict:
        """Compile C++ code and analyze the output."""
        if not self.compiler_available:
            return {'success': False, 'error': 'g++ compiler not found.'}
        if self.classifier is None:
            return {'success': False, 'error': f'Model {self.model_path} not loaded.'}

        # Create a temporary directory to compile in
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, 'temp_code.cpp')
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.write(code)

            output_exe_path = os.path.join(
                temp_dir, 'temp_exe' + ('.exe' if os.name == 'nt' else ''))

            # Compile the code
            # Added -Wno-unused to reduce noise from unused variables
            compile_command = [
                'g++', '-std=c++17', '-fdiagnostics-color=never', 
                '-Wno-unused-variable', '-Wno-unused-function',
                temp_file_path, '-o', output_exe_path
            ]
            
            try:
                process = subprocess.run(
                    compile_command,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    encoding='utf-8'
                )
                
                stdout = process.stdout
                stderr = process.stderr
                compilation_success = (process.returncode == 0)

                # Parse and classify errors
                error_messages = self._parse_output(stderr)
                classifications = self._classify_errors(error_messages)

            except subprocess.TimeoutExpired:
                return {
                    'success': True,
                    'compilation_success': False,
                    'total_errors': 1,
                    'errors': ['TimeoutError: Compilation exceeded 10 seconds.'],
                    'classifications': self._classify_errors(
                        ['TimeoutError: Compilation exceeded 10 seconds.']
                    ),
                    'summary': self._generate_summary([]),
                    'raw_stdout': '',
                    'raw_stderr': 'TimeoutError: Compilation exceeded 1a seconds.'
                }
            except Exception as e:
                return {'success': False, 'error': f'Compilation failed: {str(e)}'}

        return {
            'success': True,
            'compilation_success': compilation_success,
            'total_errors': len(error_messages),
            'errors': error_messages,
            'classifications': classifications,
            'summary': self._generate_summary(classifications),
            'raw_stdout': stdout,
            'raw_stderr': stderr
        }


# --- Python Code Analyzer ---
# (This section is UNCHANGED)

class PythonCodeErrorAnalyzer(BaseCodeErrorAnalyzer):

    def __init__(self, model_path='python_error_classifier.pkl'):
        """Initialize the analyzer with a trained model."""
        super().__init__(model_path)

    def _parse_output(self, output: str) -> List[str]:
        """
        Parse Python stderr. Returns the *last* non-empty line,
        which is typically the error message itself (e.g., "NameError: ...").
        """
        if not output:
            return []
        
        lines = output.strip().split('\n')
        
        # Find the last meaningful line
        for line in reversed(lines):
            if line.strip():
                # We only want the error, not the traceback lines
                # This is a simple heuristic: if it starts with "File ", it's traceback.
                # The *actual* error line rarely starts with "File ".
                if not line.strip().startswith('File '):
                    return [line.strip()]
        
        # Fallback: just return the last line if no better line was found
        if lines:
            return [lines[-1].strip()]
        
        return [] # No output

    def analyze(self, code: str) -> Dict:
        """Run Python code and analyze the output."""
        if self.classifier is None:
            return {'success': False, 'error': f'Model {self.model_path} not loaded.'}
        
        # Create a temporary file to run
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w', encoding='utf-8') as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Run the Python script as a subprocess
            # We use sys.executable to ensure it's the same Python
            run_command = [sys.executable, temp_file_path]

            process = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                timeout=5,
                encoding='utf-8'
            )
            
            stdout = process.stdout
            stderr = process.stderr
            execution_success = (process.returncode == 0)

            # Parse and classify errors
            error_messages = self._parse_output(stderr)
            classifications = self._classify_errors(error_messages)

        except subprocess.TimeoutExpired:
            return {
                'success': True,
                'execution_success': False,
                'total_errors': 1,
                'errors': ['TimeoutError: Code execution exceeded 5 seconds.'],
                'classifications': self._classify_errors(
                    ['TimeoutError: Code execution exceeded 5 seconds.']
                ),
                'summary': self._generate_summary([]), # No summary on timeout
                'raw_stdout': '',
                'raw_stderr': 'TimeoutError: Code execution exceeded 5 seconds.'
            }
        except Exception as e:
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}
        finally:
            # Clean up the temporary file
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return {
            'success': True,
            'execution_success': execution_success,
            'total_errors': len(error_messages),
            'errors': error_messages,
            'classifications': classifications,
            'summary': self._generate_summary(classifications),
            'raw_stdout': stdout,
            'raw_stderr': stderr
        }