
import os
import sys
import html
import json
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from datetime import datetime


# =========================================================
# 1. DASHBOARD CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MLOps Monitoring Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

from config import (
    DATA_DIR,
    DRIFT_REPORT,
    INCOMING_DIR,
    MODEL_MANIFEST,
    MODEL_PATH,
    NEW_DATA,
    PROCESSED_FILES,
    TRAINING_DATA,
    WATCHER_STATUS,
)

DATA_DIR = str(DATA_DIR)
INCOMING_DIR = str(INCOMING_DIR)
TRAINING_DATA = str(TRAINING_DATA)
NEW_DATA = str(NEW_DATA)
MODEL_PATH = str(MODEL_PATH)
DRIFT_REPORT = str(DRIFT_REPORT)
STATUS_FILE = str(WATCHER_STATUS)

DRIFT_SCRIPT = os.path.join(
    PROJECT_PATH, "src", "drift_detection.py"
)

RETRAIN_SCRIPT = os.path.join(
    PROJECT_PATH, "src", "retrain.py"
)

MODEL_CONTROL_SCRIPT = os.path.join(
    PROJECT_PATH, "model_control.py"
)

MODEL_MANIFEST = str(MODEL_MANIFEST)

os.makedirs(INCOMING_DIR, exist_ok=True)


# =========================================================
# 2. SESSION LOGS
# =========================================================

if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = []


def add_log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.execution_logs.append(
        f"[{timestamp}] [{level}] {message}"
    )


# =========================================================
# 3. DASHBOARD DESIGN
# =========================================================

st.markdown("""
<style>
:root {
    --ink: #172033;
    --muted: #64748b;
    --line: #e5eaf2;
    --surface: #ffffff;
    --canvas: #f5f7fb;
    --brand: #5146e5;
    --brand-dark: #4035c7;
    --teal: #0f9f92;
    --primary-color: #5146e5;
}

.stApp {
    background:
        radial-gradient(ellipse at 82% 0%, rgba(81, 70, 229, .06), transparent 34%),
        var(--canvas);
    color: var(--ink);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1440px !important;
    padding: 2rem 2.4rem 3rem !important;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #cbd5e1 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    color: #ffffff !important;
}

h1, h2, h3, h4 {
    color: var(--ink) !important;
    letter-spacing: -.025em;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    box-shadow: 0 5px 18px rgba(20, 32, 60, .035);
}

[data-testid="stMetric"] {
    min-height: 118px;
    padding: 16px 18px !important;
    background: #fff !important;
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(20, 32, 60, .035);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .86rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

.status-card {
    min-height: 132px;
    padding: 17px 18px;
    border: 1px solid var(--line);
    border-radius: 14px;
    background: #fff;
    box-shadow: 0 4px 14px rgba(20, 32, 60, .035);
}

.status-card-heading {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 15px;
    color: var(--muted);
    font-size: .82rem;
    font-weight: 700;
}

.status-card-icon {
    display: inline-flex;
    width: 30px;
    height: 30px;
    align-items: center;
    justify-content: center;
    border-radius: 9px;
    background: #f0efff;
    font-size: 1rem;
}

.status-card-value {
    color: var(--ink);
    font-size: 1.65rem;
    font-weight: 750;
    letter-spacing: -.035em;
    line-height: 1.1;
}

.status-card-description {
    margin-top: 8px;
    color: var(--muted);
    font-size: .73rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

div.stButton > button,
div.stDownloadButton > button {
    min-height: 44px;
    border: 1px solid #d9deea;
    border-radius: 10px !important;
    background: #fff;
    color: var(--ink);
    font-weight: 600;
    transition: all .18s ease;
}

div.stButton > button[kind="primary"] {
    border-color: var(--brand);
    background: var(--brand);
    color: #fff !important;
    box-shadow: 0 4px 10px rgba(81, 70, 229, .18);
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    border-color: var(--brand);
    color: var(--brand) !important;
    transform: translateY(-1px);
}

div.stButton > button[kind="primary"]:hover {
    border-color: var(--brand-dark);
    background: var(--brand-dark);
    color: #fff !important;
}

[data-testid="stFileUploader"],
[data-testid="stDataFrame"],
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

[data-testid="stAlert"] {
    border: 1px solid var(--line);
}

.app-hero {
    position: relative;
    overflow: hidden;
    padding: 1.7rem 2rem;
    margin: .25rem 0 1.6rem;
    border: 1px solid rgba(81, 70, 229, .16);
    border-radius: 20px;
    background: linear-gradient(115deg, #fff 5%, #f3f4ff 100%);
    box-shadow: 0 10px 28px rgba(45, 50, 100, .06);
}

.app-hero:after {
    content: "";
    position: absolute;
    width: 190px;
    height: 190px;
    right: -55px;
    top: -85px;
    border: 28px solid rgba(81, 70, 229, .07);
    border-radius: 50%;
}

.hero-kicker {
    margin: 0 0 .45rem;
    color: var(--brand);
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.hero-title {
    margin: 0;
    color: var(--ink);
    font-size: clamp(1.7rem, 3vw, 2.35rem);
    font-weight: 750;
    letter-spacing: -.04em;
}

.hero-copy {
    margin: .5rem 0 0;
    color: var(--muted);
    font-size: 1rem;
}

.log-terminal {
    min-height: 220px;
    max-height: 420px;
    overflow-y: auto;
    padding: 18px;
    border: 1px solid #263247;
    border-radius: 12px;
    background: #111827;
    color: #9de4c8;
    font-family: Consolas, monospace;
    font-size: 12px;
    line-height: 1.75;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

hr {
    border-color: var(--line);
}

@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1.15rem 1rem 2rem !important;
    }

    .app-hero {
        padding: 1.3rem 1.25rem;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def load_csv_for_dashboard(path):
    """Load CSV file for dashboard analytics."""

    if not os.path.isfile(path):
        return None

    try:
        return pd.read_csv(path)

    except Exception as exc:
        add_log(f"Could not read {path}: {exc}", "ERROR")
        return None


def file_status(path):
    return os.path.isfile(path)


def show_latest_drift_report(title="## 📊 Latest Drift Report"):
    with st.container(border=True):
        st.markdown(title)

        if not file_status(DRIFT_REPORT):
            st.info("Run drift detection to generate the report.")
            return

        try:
            with open(DRIFT_REPORT, "rb") as file:
                report_html = file.read()

            modified = datetime.fromtimestamp(
                os.path.getmtime(DRIFT_REPORT)
            )

            st.success("Latest drift report is available.")
            st.caption(
                "Generated: " + modified.strftime("%Y-%m-%d %H:%M:%S")
            )
            st.download_button(
                "📥 Download Drift Report",
                data=report_html,
                file_name="drift_report.html",
                mime="text/html",
                use_container_width=True
            )
            st.markdown("### Report Preview")
            components.html(
                report_html.decode("utf-8"),
                height=750,
                scrolling=True
            )
        except (OSError, UnicodeDecodeError) as exc:
            st.error(f"Could not display the latest drift report: {exc}")


def get_numeric_feature_columns(reference_df, incoming_df):
    """Find common numeric feature columns."""

    if reference_df is None or incoming_df is None:
        return []

    common = [
        col for col in reference_df.columns
        if col in incoming_df.columns
        and col.lower() != "target"
    ]

    return [
        col for col in common
        if pd.api.types.is_numeric_dtype(reference_df[col])
        and pd.api.types.is_numeric_dtype(incoming_df[col])
    ]


def make_feature_histogram(
    reference_df,
    incoming_df,
    feature,
    bins=10
):
    """Compare feature distributions using common bins."""

    old_values = pd.to_numeric(
        reference_df[feature], errors="coerce"
    ).dropna()

    new_values = pd.to_numeric(
        incoming_df[feature], errors="coerce"
    ).dropna()

    combined = pd.concat(
        [old_values, new_values],
        ignore_index=True
    )

    if combined.empty:
        return None

    low = combined.min()
    high = combined.max()

    if low == high:
        return pd.DataFrame(
            {
                "Reference data": [len(old_values)],
                "Incoming data": [len(new_values)]
            },
            index=[str(low)]
        )

    edges = np.linspace(low, high, bins + 1)

    old_counts = (
        pd.cut(
            old_values,
            bins=edges,
            include_lowest=True
        )
        .value_counts()
        .sort_index()
    )

    new_counts = (
        pd.cut(
            new_values,
            bins=edges,
            include_lowest=True
        )
        .value_counts()
        .sort_index()
    )

    result = pd.DataFrame({
        "Reference data": old_counts.to_numpy(),
        "Incoming data": new_counts.to_numpy()
    })

    result.index = [
        str(interval) for interval in old_counts.index
    ]

    return result


def read_watcher_status():

    if not os.path.exists(STATUS_FILE):
        return "⚠️ Watcher status file not found."

    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as file:
            return file.read().strip() or "Status file is empty."

    except OSError as exc:
        return f"Could not read watcher status: {exc}"


def get_latest_uploaded_file():

    if not os.path.isdir(INCOMING_DIR):
        return None

    files = [
        os.path.join(INCOMING_DIR, name)
        for name in os.listdir(INCOMING_DIR)
        if name.lower().endswith(".csv")
    ]

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def mark_upload_for_direct_processing(file_path):
    """Prevent the watcher duplicating work started by the dashboard."""
    file_key = os.path.abspath(file_path)
    processed = set()

    if os.path.isfile(PROCESSED_FILES):
        try:
            with open(PROCESSED_FILES, "r", encoding="utf-8") as file:
                values = json.load(file)
            if isinstance(values, list):
                processed.update(values)
        except json.JSONDecodeError:
            add_log("Watcher processed-file registry is invalid.", "WARNING")
        except OSError as exc:
            add_log(f"Could not read watcher registry: {exc}", "WARNING")

    processed.add(file_key)
    temporary_path = PROCESSED_FILES.with_suffix(".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(sorted(processed), file, indent=2)
        temporary_path.replace(PROCESSED_FILES)
    except OSError as exc:
        add_log(f"Could not update watcher registry: {exc}", "WARNING")


def set_watcher_status(message):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as file:
            file.write(message)
    except OSError as exc:
        add_log(f"Could not update workflow status: {exc}", "WARNING")


def show_metric_card(title, value, icon, description):
    safe_title = html.escape(str(title))
    safe_value = html.escape(str(value))
    safe_icon = html.escape(str(icon))
    safe_description = html.escape(str(description))

    st.markdown(
        f"""
        <div class="status-card">
          <div class="status-card-heading">
            <span class="status-card-icon">{safe_icon}</span>
            <span>{safe_title}</span>
          </div>
          <div class="status-card-value">{safe_value}</div>
          <div class="status-card-description">{safe_description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_logs():

    with st.container(border=True):

        st.markdown("## 🖥️ Execution Logs")

        st.caption(
            "View pipeline execution messages, warnings and errors."
        )

        logs = st.session_state.execution_logs

        if logs:
            log_text = "\n".join(logs[-300:])

        else:
            log_text = (
                "[SYSTEM] Waiting for pipeline execution...\n"
                "[INFO] Run drift detection or retraining to see logs."
            )

        safe_log_text = html.escape(log_text)

        st.markdown(
            f'<div class="log-terminal">{safe_log_text}</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "📥 Download Logs",
                data=log_text,
                file_name="mlops_execution_logs.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:

            if st.button(
                "🗑️ Clear Logs",
                use_container_width=True
            ):

                st.session_state.execution_logs = []
                st.rerun()


def run_script(script_path, task_name):

    with st.container(border=True):

        st.markdown(f"## ⚙️ {task_name}")

        if not os.path.isfile(script_path):

            message = f"Script not found: {script_path}"

            st.error(message)
            add_log(message, "ERROR")
            return False

        add_log(f"Started: {task_name}")

        st.info(f"⏳ Running {task_name}...")

        try:

            result = subprocess.run(
                [sys.executable, "-u", script_path],
                cwd=PROJECT_PATH,
                capture_output=True,
                text=True,
                timeout=600,
                errors="replace"
            )

            if result.stdout.strip():

                add_log(
                    "Process output:\n" + result.stdout.strip(),
                    "OUTPUT"
                )

            if result.stderr.strip():

                add_log(
                    "Additional messages:\n" + result.stderr.strip(),
                    "WARNING" if result.returncode == 0 else "ERROR"
                )

            if result.returncode == 0:

                st.success(f"✅ {task_name} completed.")

                add_log(
                    f"{task_name} completed successfully.",
                    "SUCCESS"
                )

            else:

                st.error(
                    f"❌ {task_name} failed. "
                    f"Exit code: {result.returncode}"
                )

                add_log(
                    f"{task_name} failed with exit code "
                    f"{result.returncode}.",
                    "ERROR"
                )

            with st.expander(
                "📋 View Full Execution Output",
                expanded=True
            ):

                output_text = (
                    result.stdout.strip()
                    or "No standard output."
                )

                if result.stderr.strip():

                    output_text += (
                        "\n\nADDITIONAL MESSAGES / ERRORS:\n"
                        + result.stderr.strip()
                    )

                st.code(output_text, language="text")

            return result.returncode == 0

        except subprocess.TimeoutExpired:

            message = f"{task_name} timed out after 600 seconds."

            st.error(message)
            add_log(message, "ERROR")
            return False

        except Exception as exc:

            message = f"Could not execute {task_name}: {exc}"

            st.error(message)
            add_log(message, "ERROR")
            return False


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("## ◈ MLOps Studio")

    st.caption("MODEL OPERATIONS")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Import CSV",
            "Drift Detection",
            "Model Retraining",
            "Reports & Logs"
        ]
    )

    st.divider()

    st.markdown("### 📂 System Files")

    st.write(
        "🟢 Model available"
        if file_status(MODEL_PATH)
        else "🔴 Model missing"
    )

    st.write(
        "🟢 Report available"
        if file_status(DRIFT_REPORT)
        else "🟡 Report pending"
    )

    st.write(
        "🟢 Reference data available"
        if file_status(TRAINING_DATA)
        else "🔴 Reference data missing"
    )

    st.divider()

    st.caption("Automated Model Retraining System")


# =========================================================
# 6. DASHBOARD HEADER
# =========================================================

st.markdown(
    """
    <section class="app-hero">
      <p class="hero-kicker">Model operations platform</p>
      <h1 class="hero-title">MLOps Monitoring Center</h1>
      <p class="hero-copy">Monitor incoming data, detect drift, and manage model lifecycle from one workspace.</p>
    </section>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 7. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.subheader("📌 System Overview")

    st.caption(
        "A complete view of your model monitoring pipeline."
    )

    # -----------------------------------------------------
    # SYSTEM STATUS CARDS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:

        show_metric_card(
            "Reference Data",
            "Ready" if file_status(TRAINING_DATA) else "Missing",
            "📁",
            "data/training_data.csv"
        )

    with c2:

        show_metric_card(
            "Incoming Data",
            "Ready" if file_status(NEW_DATA) else "Missing",
            "📥",
            "data/new_data.csv"
        )

    with c3:

        show_metric_card(
            "ML Model",
            "Ready" if file_status(MODEL_PATH) else "Missing",
            "🧠",
            "models/model.pkl"
        )

    with c4:

        show_metric_card(
            "Drift Report",
            "Ready" if file_status(DRIFT_REPORT) else "Pending",
            "📊",
            "data/drift_report.html"
        )

    # -----------------------------------------------------
    # DATASET ANALYTICS
    # -----------------------------------------------------

    with st.container(border=True):

        st.markdown("## 📈 Dataset Analytics")

        st.caption(
            "Live comparisons using reference and incoming CSV files."
        )

        reference_df = load_csv_for_dashboard(TRAINING_DATA)
        incoming_df = load_csv_for_dashboard(NEW_DATA)

        if reference_df is None:

            st.warning("Reference dataset is missing or unreadable.")

        elif incoming_df is None:

            st.info(
                "Upload a CSV and select 'Use Latest CSV as Incoming Data' "
                "to display dataset comparisons."
            )

            st.markdown("### Reference Dataset Summary")

            st.dataframe(
                pd.DataFrame({
                    "Rows": [len(reference_df)],
                    "Columns": [len(reference_df.columns)],
                    "Missing Values": [
                        int(reference_df.isna().sum().sum())
                    ]
                }, index=["Reference"]),
                use_container_width=True
            )

        else:

            common_columns = [
                col for col in reference_df.columns
                if col in incoming_df.columns
            ]

            # Dataset size chart

            chart_left, chart_right = st.columns(2)

            with chart_left:

                st.markdown("### 📊 Dataset Size Comparison")

                size_data = pd.DataFrame(
                    {
                        "Rows": [
                            len(reference_df),
                            len(incoming_df)
                        ],
                        "Columns": [
                            len(reference_df.columns),
                            len(incoming_df.columns)
                        ]
                    },
                    index=["Reference", "Incoming"]
                )

                st.bar_chart(
                    size_data,
                    use_container_width=True
                )

            # Missing values chart

            with chart_right:

                st.markdown("### ⚠️ Missing Values Comparison")

                missing_data = pd.DataFrame(
                    {
                        "Missing Values": [
                            int(reference_df.isna().sum().sum()),
                            int(incoming_df.isna().sum().sum())
                        ]
                    },
                    index=["Reference", "Incoming"]
                )

                st.bar_chart(
                    missing_data,
                    use_container_width=True
                )

            # Feature distribution chart

            st.markdown("### 📉 Feature Distribution Comparison")

            numeric_features = get_numeric_feature_columns(
                reference_df,
                incoming_df
            )

            if numeric_features:

                selected_feature = st.selectbox(
                    "Select a numeric feature",
                    numeric_features,
                    key="overview_feature_distribution"
                )

                histogram_data = make_feature_histogram(
                    reference_df,
                    incoming_df,
                    selected_feature
                )

                if histogram_data is not None:

                    st.caption(
                        "Comparison of value distributions across "
                        "the same value ranges."
                    )

                    st.bar_chart(
                        histogram_data,
                        use_container_width=True
                    )

            elif common_columns:

                st.info(
                    "Common columns exist, but no common numeric "
                    "feature columns are available."
                )

            else:

                st.warning(
                    "No matching feature columns found. Upload a CSV "
                    "with compatible feature names."
                )

    # -----------------------------------------------------
    # QUICK ACTIONS AND MONITORING
    # -----------------------------------------------------

    left, right = st.columns([1, 1], gap="medium")

    with left:

        with st.container(border=True):

            st.markdown("## 🔧 Quick Actions")

            st.caption("Execute your existing pipeline scripts.")

            if st.button(
                "🔍 Run Drift Detection",
                type="primary",
                use_container_width=True
            ):

                run_script(
                    DRIFT_SCRIPT,
                    "Data Drift Detection"
                )

            if st.button(
                "🔄 Run Retraining Workflow",
                use_container_width=True
            ):

                run_script(
                    RETRAIN_SCRIPT,
                    "Model Retraining Workflow"
                )

            if st.button(
                "📋 Open Reports & Logs",
                use_container_width=True
            ):

                st.info(
                    "Select Reports & Logs from the sidebar."
                )

    with right:

        with st.container(border=True):

            st.markdown("## 👁️ Monitoring Status")

            st.caption("Status reported by watcher.py.")

            st.write(read_watcher_status())

            if st.button(
                "🔄 Refresh Status",
                use_container_width=True
            ):

                st.rerun()

    # -----------------------------------------------------
    # PIPELINE WORKFLOW
    # -----------------------------------------------------

    with st.container(border=True):

        st.markdown("## 🔗 Pipeline Workflow")

        a, b, c = st.columns(3, gap="medium")

        with a:

            st.markdown("### 📥 01. Import")

            st.write(
                "Upload a CSV and inspect its columns and values."
            )

            st.caption("Input: CSV dataset")

        with b:

            st.markdown("### 🔍 02. Detect")

            st.write(
                "Compare matching features against reference data."
            )

            st.caption("Tool: Evidently")

        with c:

            st.markdown("### 🧠 03. Retrain")

            st.write(
                "Execute the configured retraining workflow."
            )

            st.caption("Script: src/retrain.py")

    show_latest_drift_report()
    show_logs()


# =========================================================
# 8. IMPORT CSV PAGE
# =========================================================

elif page == "Import CSV":

    st.subheader("📥 Import Incoming Dataset")

    with st.container(border=True):

        st.markdown("## 📂 Upload CSV File")

        st.caption(
            "Choose a CSV file to preview it. It is saved automatically "
            "and drift detection starts as soon as the upload completes. "
            "Feature headers may differ; compatible columns are paired by type and order."
        )

        uploaded_file = st.file_uploader(
            "Select a CSV file",
            type=["csv"],
            key="csv_upload"
        )

        if uploaded_file is not None:

            try:

                preview_df = pd.read_csv(uploaded_file)

                m1, m2, m3 = st.columns(3)

                m1.metric("📄 Rows", len(preview_df))

                m2.metric("📊 Columns", len(preview_df.columns))

                m3.metric(
                    "⚠️ Missing Values",
                    int(preview_df.isna().sum().sum())
                )

                st.markdown("### 👁️ Dataset Preview")

                st.dataframe(
                    preview_df.head(100),
                    use_container_width=True
                )

                st.markdown("### 🧾 Column Information")

                column_info = pd.DataFrame({
                    "Column": preview_df.columns,
                    "Data Type": [
                        str(dtype) for dtype in preview_df.dtypes
                    ],
                    "Missing Values": [
                        int(preview_df[col].isna().sum())
                        for col in preview_df.columns
                    ]
                })

                st.dataframe(
                    column_info,
                    use_container_width=True
                )

                upload_token = (
                    f"{uploaded_file.file_id}:{uploaded_file.name}"
                )

                if st.session_state.get("last_auto_upload") != upload_token:
                    st.session_state.last_auto_upload = upload_token

                    try:
                        safe_name = os.path.basename(uploaded_file.name)
                        save_path = os.path.join(INCOMING_DIR, safe_name)
                        temporary_upload = save_path + ".uploading"

                        uploaded_file.seek(0)
                        with open(temporary_upload, "wb") as file:
                            file.write(uploaded_file.getbuffer())

                        mark_upload_for_direct_processing(save_path)
                        os.replace(temporary_upload, save_path)
                        preview_df.to_csv(NEW_DATA, index=False)

                        add_log(
                            f"CSV saved and staged for drift detection: {safe_name}",
                            "SUCCESS"
                        )

                        st.success(
                            f"{safe_name} was saved. Drift detection and "
                            "automatic retraining (if drift is found) are starting."
                        )

                        workflow_succeeded = run_script(
                            RETRAIN_SCRIPT,
                            "Automatic Drift Detection and Retraining"
                        )
                        set_watcher_status(
                            "SUCCESS: Uploaded CSV analyzed. "
                            "See dashboard output for the drift result."
                            if workflow_succeeded
                            else "ERROR: Automatic drift workflow failed. "
                            "See dashboard output."
                        )
                    except OSError as exc:
                        st.error(f"Could not save uploaded CSV: {exc}")
                        add_log(f"CSV save failed: {exc}", "ERROR")

            except Exception as exc:

                st.error(f"Could not read this CSV: {exc}")

                add_log(
                    f"CSV import failed: {exc}",
                    "ERROR"
                )

    show_latest_drift_report()

    # Latest uploaded dataset

    with st.container(border=True):

        st.markdown("## 📌 Latest Uploaded Dataset")

        latest_file = get_latest_uploaded_file()

        if latest_file:

            st.write(
                "**Filename:**",
                os.path.basename(latest_file)
            )

            modified = datetime.fromtimestamp(
                os.path.getmtime(latest_file)
            )

            st.write(
                "**Last modified:**",
                modified.strftime("%Y-%m-%d %H:%M:%S")
            )

            if st.button(
                "📤 Use Latest CSV as Incoming Data",
                type="primary",
                use_container_width=True
            ):

                try:

                    df = pd.read_csv(latest_file)

                    os.makedirs(DATA_DIR, exist_ok=True)

                    df.to_csv(NEW_DATA, index=False)

                    add_log(
                        f"Copied {os.path.basename(latest_file)} "
                        "to data/new_data.csv",
                        "SUCCESS"
                    )

                    st.success(
                        "Incoming dataset updated. Starting drift detection "
                        "and automatic retraining if drift is detected."
                    )
                    workflow_succeeded = run_script(
                        RETRAIN_SCRIPT,
                        "Automatic Drift Detection and Retraining"
                    )
                    set_watcher_status(
                        "SUCCESS: Incoming dataset analyzed. "
                        "See dashboard output for the drift result."
                        if workflow_succeeded
                        else "ERROR: Automatic drift workflow failed. "
                        "See dashboard output."
                    )

                except Exception as exc:

                    st.error(
                        f"Could not update incoming data: {exc}"
                    )

                    add_log(
                        f"Incoming data update failed: {exc}",
                        "ERROR"
                    )

        else:

            st.info(
                "Upload and save a CSV file to see it here."
            )

    show_logs()


# =========================================================
# 9. DRIFT DETECTION PAGE
# =========================================================

elif page == "Drift Detection":

    st.subheader("🔍 Data Drift Detection")

    left, right = st.columns(2)

    # Reference dataset

    with left:

        with st.container(border=True):

            st.markdown("## 📁 Reference Dataset")

            st.write("data/training_data.csv")

            if file_status(TRAINING_DATA):

                st.success("Reference dataset available.")

                try:

                    df = pd.read_csv(TRAINING_DATA)

                    x, y = st.columns(2)

                    x.metric("Rows", len(df))
                    y.metric("Columns", len(df.columns))

                except Exception as exc:

                    st.error(
                        f"Cannot read reference dataset: {exc}"
                    )

            else:

                st.error("Reference dataset not found.")

    # Incoming dataset

    with right:

        with st.container(border=True):

            st.markdown("## 📥 Incoming Dataset")

            st.write("data/new_data.csv")

            if file_status(NEW_DATA):

                st.success("Incoming dataset available.")

                try:

                    df = pd.read_csv(NEW_DATA)

                    x, y = st.columns(2)

                    x.metric("Rows", len(df))
                    y.metric("Columns", len(df.columns))

                except Exception as exc:

                    st.error(
                        f"Cannot read incoming dataset: {exc}"
                    )

            else:

                st.warning(
                    "Upload a CSV and set it as incoming data."
                )

    # Run drift detection

    with st.container(border=True):

        st.markdown("## ⚙️ Run Drift Detection")

        st.write(
            "The configured script compares compatible features "
            "and generates an Evidently HTML report."
        )

        if st.button(
            "🔍 Start Drift Detection",
            type="primary",
            use_container_width=True
        ):

            run_script(
                DRIFT_SCRIPT,
                "Data Drift Detection"
            )

    show_latest_drift_report()

    show_logs()


# =========================================================
# 10. MODEL RETRAINING PAGE
# =========================================================

elif page == "Model Retraining":

    st.subheader("🧠 Model Retraining Center")

    c1, c2 = st.columns(2)

    with c1:

        show_metric_card(
            "Saved Model",
            "Available" if file_status(MODEL_PATH) else "Missing",
            "🧠",
            "models/model.pkl"
        )

    with c2:

        show_metric_card(
            "Incoming Dataset",
            "Available" if file_status(NEW_DATA) else "Missing",
            "📥",
            "data/new_data.csv"
        )

    with st.container(border=True):

        st.markdown("## 🔄 Retraining Workflow")

        st.write(
            "Run the existing retraining script. The actual training "
            "logic and retraining conditions remain controlled by "
            "src/retrain.py."
        )

        if st.button(
            "🚀 Run Retraining Workflow",
            type="primary",
            use_container_width=True
        ):

            run_script(
                RETRAIN_SCRIPT,
                "Model Retraining Workflow"
            )

    with st.container(border=True):

        st.markdown("## 📦 Model Approval and Rollback")

        if file_status(MODEL_MANIFEST):
            try:
                with open(MODEL_MANIFEST, "r", encoding="utf-8") as file:
                    manifest = json.load(file)

                st.write("**Status:**", manifest.get("status", "unknown"))
                st.write("**Candidate accuracy:**", manifest.get("accuracy", "N/A"))
                st.write("**Training source:**", manifest.get("training_source", "N/A"))
                st.write("**Created:**", manifest.get("created_at", "N/A"))

            except Exception as exc:
                st.error(f"Could not read model manifest: {exc}")
        else:
            st.info("No retraining candidate manifest is available.")

        approve_column, rollback_column = st.columns(2)

        with approve_column:
            if st.button("✅ Approve Candidate", use_container_width=True):
                result = subprocess.run(
                    [sys.executable, MODEL_CONTROL_SCRIPT, "approve"],
                    cwd=PROJECT_PATH,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    errors="replace"
                )
                if result.returncode == 0:
                    st.success(result.stdout.strip())
                    st.rerun()
                else:
                    st.error(result.stderr.strip() or "Approval failed.")

        with rollback_column:
            if st.button("↩️ Roll Back Model", use_container_width=True):
                result = subprocess.run(
                    [sys.executable, MODEL_CONTROL_SCRIPT, "rollback"],
                    cwd=PROJECT_PATH,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    errors="replace"
                )
                if result.returncode == 0:
                    st.success(result.stdout.strip())
                    st.rerun()
                else:
                    st.error(result.stderr.strip() or "Rollback failed.")

    show_logs()


# =========================================================
# 11. REPORTS AND LOGS PAGE
# =========================================================

elif page == "Reports & Logs":

    st.subheader("📊 Reports & Execution Logs")

    c1, c2 = st.columns(2)

    with c1:
        show_latest_drift_report("## 📈 Drift Report")

    # Watcher status

    with c2:

        with st.container(border=True):

            st.markdown("## 👁️ Watcher Status")

            st.write(read_watcher_status())

            if st.button(
                "🔄 Refresh Watcher Status",
                use_container_width=True
            ):

                st.rerun()

    show_logs()


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.caption(
    "📊 MLOps Monitoring Center | "
    "🔍 Data Drift | "
    "🧠 Model Retraining | "
    "📈 Evidently | "
    "🖥️ Execution Logs"
)
