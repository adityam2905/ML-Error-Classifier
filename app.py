"""
Streamlit App for C++ and Python Error Classification.

The analysis logic lives in 'analyzers.py' and the ML logic in
'error_classifier.py'. NLTK data is downloaded when 'error_classifier.py'
is imported (via 'analyzers.py').
"""
import streamlit as st
import os
import io

st.set_page_config(page_title="Code Error Classifier", layout="wide")

# This import chain also triggers the NLTK download check
# (app.py -> analyzers.py -> error_classifier.py)
try:
    from analyzers import CppCodeErrorAnalyzer, PythonCodeErrorAnalyzer
except ImportError as e:
    st.error(
        f"Failed to import analyzers. Is 'error_classifier.py' missing? Error: {e}")
    st.stop()

# ======================== THEME STYLING ==========================
st.markdown("""
<style>
/* Base */
html, body, [class*="css"], .block-container {
    background-color: #000000 !important;
    color: #FFFFFF !important; /* Default text white */
    font-family: 'Poppins', sans-serif;
}

/* All other text elements (labels, paragraphs, etc.) */
label, span, p, div {
    color: #FFFFFF !important;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #FF3B3B !important;
}

/* Sidebar - Kept clean */
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
    background-color: #000000 !important;
    border-right: 1px solid #660000;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    border: 1px solid transparent !important;
    background-color: #000000 !important;
}

/* Buttons */
.stButton>button {
    background-color: #B30000 !important;
    color: #FFFFFF !important;
    font-weight: 600;
    border-radius: 6px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #FF0000 !important;
}
        
/* Top header bar */
header[data-testid="stHeader"], .stAppHeader {
    background-color: #111111 !important;
    border-bottom: 1px solid #550000 !important;
}

/* === THE READABILITY FIX (SCOPED TO MAIN CONTAINER) === */

/* Target only the main app container, NOT the sidebar */
[data-testid="stAppViewContainer"] {

    /* Input Boxes: Text Area & Text Input */
    .stTextArea textarea, .stTextInput input {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #880000 !important;
        border-radius: 6px;
    }

    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #111111 !important;
        border: 1px solid #880000 !important;
        border-radius: 6px;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        color: #FFFFFF !important;
    }

    /* --- FILE UPLOADER FIX --- */
    
    /* Kill all white/light backgrounds in file uploader */
    .stFileUploader, 
    .stFileUploader > div,
    .stFileUploader section,
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploadDropzone"] > div,
    [data-testid="stFileUploadDropzone"] section {
        background-color: #2B2B2B !important;
        border: none !important;
    }
    
    /* Main dropzone styling */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #666666 !important;
        border-radius: 8px !important;
        padding: 2rem !important;
    }
    
    /* All text inside file uploader */
    .stFileUploader *,
    [data-testid="stFileUploadDropzone"] *,
    [data-testid="stFileUploadDropzoneInstructions"] * {
        color: #DDDDDD !important;
        background-color: transparent !important;
    }
    
    /* Label above uploader */
    .stFileUploader > label {
        color: #FFFFFF !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Browse files button */
    .stFileUploader button,
    [data-testid="stFileUploadDropzone"] button {
        background-color: #8B0000 !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stFileUploader button:hover,
    [data-testid="stFileUploadDropzone"] button:hover {
        background-color: #C00000 !important;
    }
    
    /* Uploaded file display */
    [data-testid="stFileUploaderFile"] {
        background-color: #1a1a1a !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
        border-radius: 4px !important;
        padding: 0.5rem !important;
    }
    /* --- END FILE UPLOADER FIX --- */
}
/* === END FIX === */

/* Selectbox - dropdown menu (popover is global) */
[data-baseweb="popover"] li {
    background-color: #111111 !important;
    color: #FFFFFF !important;
}
[data-baseweb="popover"] li:hover {
    background-color: #550000 !important;
}

/* === NUCLEAR OPTION - KILL ALL WHITE BACKGROUNDS === */

/* Force EVERYTHING to dark */
*, *::before, *::after {
    border-color: #333333 !important;
}

/* All divs and containers */
div, section, article, aside, main {
    background-color: transparent !important;
}

/* Specific white box killers */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
.element-container,
.stMarkdown,
div[class*="css"] {
    background-color: transparent !important;
}

/* Expanders - NO BLEEDING */
.stExpander {
    background-color: #1a1a1a !important;
    border: 1px solid #444444 !important;
    border-radius: 6px !important;
    margin: 0.5rem 0 !important;
}

/* Expander header */
.stExpander summary {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
    padding: 0.75rem !important;
    border: none !important;
}

/* Expander header on hover */
.stExpander summary:hover {
    background-color: #2a2a2a !important;
}

/* Expander content area - NO BLEEDING */
.stExpander > div[role="region"] {
    background-color: #0a0a0a !important;
    padding: 1rem !important;
    border: none !important;
    margin: 0 !important;
}

/* Code blocks - ZERO BORDERS */
.stCode, .stCodeBlock, pre, code {
    background-color: #0a0a0a !important;
    color: #FFAAAA !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 1rem !important;
    margin: 0 !important;
}

/* Code block pre and code elements */
.stCode pre, .stCode code,
.stCodeBlock pre, .stCodeBlock code {
    background-color: transparent !important;
    color: #FFAAAA !important;
    border: none !important;
}

/* === DATAFRAMES - COMPLETE DARK MODE === */
.dataframe, 
.stDataFrame, 
[data-testid="stDataFrame"],
table,
.row-widget.stDataFrame,
div[data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"] {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
    border: 1px solid #333333 !important;
}

/* Dataframe wrapper - the actual container */
div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] > div,
div[data-testid="stDataFrame"] iframe {
    background-color: #1a1a1a !important;
}

/* Dataframe container */
[data-testid="stDataFrame"] > div,
.stDataFrame > div {
    background-color: #1a1a1a !important;
}

/* Style the actual table element */
.stDataFrame table,
[data-testid="stDataFrame"] table {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
}

/* Dataframe headers */
.dataframe thead th, 
.stDataFrame thead th,
[data-testid="stDataFrame"] thead th,
table thead th,
.stDataFrame table thead th {
    background-color: #2a2a2a !important;
    color: #FFFFFF !important;
    border: 1px solid #444444 !important;
    font-weight: 600 !important;
}

/* Dataframe cells */
.dataframe tbody td,
.stDataFrame tbody td,
[data-testid="stDataFrame"] tbody td,
table tbody td,
.stDataFrame table tbody td {
    background-color: #1a1a1a !important;
    color: #DDDDDD !important;
    border: 1px solid #333333 !important;
}

/* Dataframe rows */
.dataframe tbody tr,
.stDataFrame tbody tr,
[data-testid="stDataFrame"] tbody tr,
table tbody tr,
.stDataFrame table tbody tr {
    background-color: #1a1a1a !important;
}

/* Dataframe alternating rows */
.dataframe tbody tr:nth-child(even),
.stDataFrame tbody tr:nth-child(even),
[data-testid="stDataFrame"] tbody tr:nth-child(even),
table tbody tr:nth-child(even),
.stDataFrame table tbody tr:nth-child(even) {
    background-color: #151515 !important;
}

/* Dataframe hover */
.dataframe tbody tr:hover,
.stDataFrame tbody tr:hover,
[data-testid="stDataFrame"] tbody tr:hover,
table tbody tr:hover,
.stDataFrame table tbody tr:hover {
    background-color: #252525 !important;
}

/* Index column (the 0, 1, 2 numbers) */
.dataframe tbody th,
.stDataFrame tbody th,
[data-testid="stDataFrame"] tbody th,
.stDataFrame table tbody th {
    background-color: #2a2a2a !important;
    color: #AAAAAA !important;
    border: 1px solid #444444 !important;
}

/* === METRIC BOXES - DARK === */
.stMetric, [data-testid="stMetric"] {
    background-color: #1a1a1a !important;
    border: 1px solid #444444 !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}

.stMetric label, 
.stMetric div,
[data-testid="stMetric"] label,
[data-testid="stMetric"] div {
    color: #FFFFFF !important;
    background-color: transparent !important;
}

/* === TOOLTIPS AND POPOVERS - DARK === */
.stTooltipIcon,
[data-testid="stTooltipHoverTarget"],
div[role="tooltip"] {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
    border: 1px solid #444444 !important;
}

/* === INFO/WARNING/ERROR BOXES === */
.stAlert, [data-testid="stAlert"] {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
    border: 1px solid #444444 !important;
}

/* === MARKDOWN CONTENT === */
.stMarkdown, .stMarkdown * {
    color: #FFFFFF !important;
}

/* === ANY INLINE STYLES WITH WHITE BACKGROUNDS === */
[style*="background: white"],
[style*="background-color: white"],
[style*="background: rgb(255, 255, 255)"],
[style*="background-color: rgb(255, 255, 255)"],
[style*="background: #fff"],
[style*="background-color: #fff"],
[style*="background: #FFF"],
[style*="background-color: #FFF"] {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
}

/* Dataframes */
.dataframe {
    background-color: #000000 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb {
    background: #550000;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
# ================================================================


@st.cache_resource
def load_cpp_analyzer():
    model_path = "cpp_error_classifier.pkl"
    if not os.path.exists(model_path):
        st.error("C++ model file not found! Run `python mlbec3.py` to train.")
        return None
    try:
        analyzer = CppCodeErrorAnalyzer(model_path=model_path)
        if analyzer.classifier is None:
            st.error(
                "Failed to load C++ model. Please retrain with `python mlbec3.py`.")
            return None
        return analyzer
    except Exception as e:
        st.error(f"Error initializing CppCodeErrorAnalyzer: {e}")
        return None


@st.cache_resource
def load_python_analyzer():
    model_path = "python_error_classifier.pkl"
    if not os.path.exists(model_path):
        st.error("Python model file not found! Run `python mlbec3.py` to train.")
        return None
    try:
        analyzer = PythonCodeErrorAnalyzer(model_path=model_path)
        if analyzer.classifier is None:
            st.error(
                "Failed to load Python model. Please retrain with `python mlbec3.py`.")
            return None
        return analyzer
    except Exception as e:
        st.error(f"Error initializing PythonCodeErrorAnalyzer: {e}")
        return None


def display_results(results: dict, language: str):
    """Unified result display for both analyzers"""
    if not results["success"]:
        st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        return

    success_key = "compilation_success" if language == "C++" else "execution_success"
    success_label = "Compilation" if language == "C++" else "Execution"

    if results[success_key]:
        st.success(f"{success_label} Successful!")
        if results["total_errors"] == 0:
            st.balloons()
            st.markdown("### No errors found!")
        else:
            st.info(f"Found {results['total_errors']} warnings.")
    else:
        st.error(f"{success_label} Failed.")
        st.markdown(
            f"Found **{results['total_errors']}** error(s)/warning(s).")

    st.subheader("Analysis Summary")
    summary = results.get('summary', {})
    if not summary:
        st.write("No errors to summarize.")
        return

    cols = st.columns(len(summary))
    for i, (err_type, count) in enumerate(summary.items()):
        cols[i].metric(err_type.capitalize(), count)

    st.subheader("Classified Errors")
    classifications = results.get('classifications', [])
    if not classifications:
        st.write("No classified errors to display.")

    for item in classifications:
        pred_type = item['predicted_type'].replace('_', ' ').title()
        conf = item['confidence']
        with st.expander(f"{pred_type} (Confidence: {conf:.1%})"):
            st.markdown(f"**Predicted Type:** {pred_type}")
            st.markdown(f"**Confidence:** `{conf:.4f}`")
            st.markdown("**Original Error Message:**")
            st.code(item['error_message'], language="bash")
            probs = item.get('all_probabilities', {})
            if probs:
                prob_df = {"Error Type": [k.replace('_', ' ').title() for k in probs.keys()],
                           "Probability": [f"{v:.2%}" for v in probs.values()]}
                st.dataframe(prob_df, width="stretch")
            else:
                st.write("No probability distribution available.")

    with st.expander("Show Raw Compiler/Interpreter Output"):
        st.subheader("Raw STDOUT")
        st.text(results.get('raw_stdout', 'N/A'))
        st.subheader("Raw STDERR")
        st.text(results.get('raw_stderr', 'N/A'))


# --------------------------------------------------------
# MAIN STREAMLIT APP
# --------------------------------------------------------
st.title("ML Based Code Error Classifier")
st.markdown(
    "Upload a C++ or Python file to detect and classify code errors using ML.")

st.sidebar.header("Configuration")
language = st.sidebar.selectbox("Select Language", ["C++", "Python"])

if language == "C++":
    analyzer = load_cpp_analyzer()
    if analyzer is None:
        st.stop()
    if not analyzer.compiler_available:
        st.error("g++ compiler not found. Please install g++.")
        st.stop()
    st.success(
        f"C++ analyzer ready (Model: {analyzer.classifier.best_model_name})")
    accept_types = ["cpp", "c", "h", "hpp", "txt"]
else:
    analyzer = load_python_analyzer()
    if analyzer is None:
        st.stop()
    st.success(
        f"Python analyzer ready (Model: {analyzer.classifier.best_model_name})")
    accept_types = ["py", "txt"]

st.header("Upload Your Code File")
uploaded = st.file_uploader(
    f"Drop your .{accept_types[0]} file here", type=accept_types)
code = ""

if uploaded:
    try:
        code = io.StringIO(uploaded.getvalue().decode("utf-8")).read()
    except UnicodeDecodeError:
        st.error("Failed to decode file. Please ensure UTF-8 encoding.")
        st.stop()
else:
    st.subheader("...or paste your code here:")
    default_code = "#include <iostream>\n\nint main() {\n    std::cout << \"Hello\" \n    return 0;\n}\n" if language == "C++" else "def hello():\n    print(\"Hello\")\n\nhello)\n"
    code = st.text_area("Code Editor", default_code,
                        height=300, key=f"code_editor_{language}")

if code:
    with st.expander("Show Code"):
        st.code(code, language="cpp" if language == "C++" else "python")
    if st.button("Analyze Code", type="primary", width="stretch"):
        with st.spinner(f"Running and classifying {language} errors..."):
            results = analyzer.analyze(code)
            display_results(results, language)
