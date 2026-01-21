import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
import json
import os

st.set_page_config(
    page_title="CSAT Prediction Dashboard",
    page_icon="CSAT",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
DATA_PATH = Path('eCommerce_Customer_support_data.csv')
PREPROCESSOR_PATH = Path('preprocessor.pkl')
MODEL_PATHS = [
    Path('ensemble_model_1_wide_shallow.h5'),
    Path('ensemble_model_2_deep_narrow.h5'),
    Path('ensemble_model_3_balanced_swish.h5'),
]
# Deterministic RNG for metrics/balanced splits
METRICS_RNG = np.random.default_rng(42)
BASE_DIR = Path(__file__).resolve().parent
METRICS_FILE = BASE_DIR / 'model_metrics.json'


def sanitize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 1:
        probs = probs[None, :]
    np.nan_to_num(probs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs /= row_sums
    return probs


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'Unique id', 'Customer Remarks', 'order_date_time', 'Customer_City',
        'Product_category', 'Item_price', 'connected_handling_time', 'Order_id',
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], axis=1)

    df['issue_reported_at'] = pd.to_datetime(df['Issue_reported at'], errors='coerce')
    df['issue_responded'] = pd.to_datetime(df['issue_responded'], errors='coerce')

    df['response_time_minutes'] = (df['issue_responded'] - df['issue_reported_at']).dt.total_seconds() / 60
    df['response_time_minutes'] = df['response_time_minutes'].clip(lower=0)
    median_response_time = df['response_time_minutes'].median()
    df['response_time_minutes'] = df['response_time_minutes'].fillna(median_response_time)

    df['hour_reported'] = df['issue_reported_at'].dt.hour
    df['day_of_week'] = df['issue_reported_at'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour_reported'] >= 9) & (df['hour_reported'] <= 17)).astype(int)
    df['month'] = df['issue_reported_at'].dt.month
    df['log_response_time'] = np.log1p(df['response_time_minutes'])

    datetime_cols = ['Issue_reported at', 'issue_reported_at', 'issue_responded', 'Survey_response_Date']
    df = df.drop(columns=[c for c in datetime_cols if c in df.columns])

    if 'Agent_name' in df.columns:
        agent_perf = df.groupby('Agent_name')['CSAT Score'].agg(['mean', 'count']).reset_index()
        agent_perf.columns = ['Agent_name', 'agent_avg_csat', 'agent_case_count']
        df = df.merge(agent_perf, on='Agent_name', how='left')

    return df


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_resource(show_spinner=False)
def load_models():
    return [load_model(p) for p in MODEL_PATHS]


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_preprocessed():
    raw = load_dataset()
    clean = clean_data(raw.copy())
    y = clean['CSAT Score'].reset_index(drop=True)
    X = clean.drop('CSAT Score', axis=1).reset_index(drop=True)
    preprocessor = load_preprocessor()
    Xp = preprocessor.transform(X)
    if hasattr(Xp, 'toarray'):
        Xp = Xp.toarray()
    Xp = np.asarray(Xp, dtype=np.float32)
    valid_mask = ~np.isnan(Xp).any(axis=1)
    Xp = Xp[valid_mask]
    y = y[valid_mask].reset_index(drop=True)
    return Xp, y


def hybrid_predict(models, X):
    m1, m2, m3 = models
    X = np.asarray(X, dtype=np.float32)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    p1 = m1.predict(X, verbose=0)
    p2 = m2.predict(X, verbose=0)
    p3 = m3.predict(X, verbose=0)
    ens = (p1 + p2 + p3) / 3.0
    pred1 = np.argmax(p1, axis=1)
    hybrid_pred = np.argmax(ens, axis=1)
    mask1 = pred1 == 0
    mask5 = pred1 == 4
    hybrid_pred = hybrid_pred.copy()
    hybrid_pred[mask1] = 0
    hybrid_pred[mask5] = 4
    hybrid_proba = ens.copy()
    hybrid_proba[mask1] = p1[mask1]
    hybrid_proba[mask5] = p1[mask5]
    hybrid_proba = sanitize_probabilities(hybrid_proba)
    return hybrid_pred + 1, hybrid_proba


def compute_balanced_indices(labels, per_class=500):
    idx_list = []
    labels = pd.Series(labels).reset_index(drop=True)
    for csat in range(1, 6):
        idx = labels[labels == csat].index.to_numpy()
        if len(idx) > per_class:
            chosen = METRICS_RNG.choice(idx, size=per_class, replace=False)
        else:
            chosen = idx
        idx_list.extend(chosen.tolist())
    return np.sort(np.asarray(idx_list, dtype=int))


@st.cache_data(show_spinner=False)
def compute_metrics_cached():
    try:
        Xp, y = get_preprocessed()
        models = load_models()
        if len(Xp) == 0:
            return {'success': False, 'error': 'No samples available'}
        # Use the full preprocessed dataset for evaluation to reflect real distribution
        X_eval = Xp
        y_eval = y.to_numpy() - 1
        preds = []
        chunk = 512
        for start in range(0, len(X_eval), chunk):
            batch = X_eval[start:start + chunk]
            p, _ = hybrid_predict(models, batch)
            preds.append(p.astype(int))
        preds_all = np.concatenate(preds) if preds else np.array([], dtype=int)
        actual = y_eval + 1
        confusion = np.zeros((5, 5), dtype=int)
        for a, p in zip(actual, preds_all):
            confusion[a - 1, p - 1] += 1
        accuracy = float(np.mean(preds_all == actual) * 100.0) if len(actual) else 0.0
        error_counts = {
            '1_to_5': int(np.sum((actual == 1) & (preds_all == 5))),
            '5_to_1': int(np.sum((actual == 5) & (preds_all == 1))),
            'other': int(np.sum((preds_all != actual) & ~(((actual == 1) & (preds_all == 5)) | ((actual == 5) & (preds_all == 1))))),
        }
        per_class_acc = []
        for cls in range(1, 6):
            mask = actual == cls
            denom = int(np.sum(mask))
            if denom == 0:
                per_class_acc.append({'class': f'CSAT {cls}', 'accuracy': None})
            else:
                per_class_acc.append({'class': f'CSAT {cls}', 'accuracy': float(np.mean(preds_all[mask] == actual[mask]) * 100.0)})
        return {
            'success': True,
            'accuracy': accuracy,
            'total_samples': int(len(actual)),
            'error_breakdown': error_counts,
            'confusion_matrix': confusion.tolist(),
            'per_class_accuracy': per_class_acc,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def load_static_metrics():
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            try:
                return float(str(x).replace('%', ''))
            except Exception:
                return None

    def _to_int(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return None

    paths = [METRICS_FILE, Path('model_metrics.json')]
    for path in paths:
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                # normalize numeric fields
                for k, v in list(raw.items()):
                    if k == "timestamp" or not isinstance(v, dict):
                        continue
                    if 'accuracy' in v and v['accuracy'] is not None:
                        v['accuracy'] = _to_float(v['accuracy'])
                    if 'errors' in v and isinstance(v['errors'], dict):
                        for ek, ev in v['errors'].items():
                            v['errors'][ek] = _to_int(ev) if ev is not None else None
                raw['__source'] = str(path.resolve())
                return raw
            except Exception:
                continue
    return None


# Sidebar navigation
st.sidebar.title("Navigation")
menu_choice = st.sidebar.selectbox(
    "Go to",
    ["Home", "Single Prediction", "Batch Testing", "Model Performance", "About"],
    index=0,
)

# Load resources once
models = load_models()
Xp, y_full = get_preprocessed()
max_index = len(Xp) - 1

# Initialize session state
if 'sample_index' not in st.session_state:
    st.session_state.sample_index = 0
if 'batch_indices' not in st.session_state:
    st.session_state.batch_indices = []
if 'rng' not in st.session_state:
    st.session_state.rng = np.random.default_rng()

if menu_choice == "Home":
    st.header("CSAT Prediction Dashboard")
    # Load metrics (may be cached)
    metrics = compute_metrics_cached()
    metrics_ready = isinstance(metrics, dict) and metrics.get('success')
    errors = metrics.get('error_breakdown') if metrics_ready else {}

    # Theme cards styling
    st.markdown("""
    <style>
    .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 8px; }
    .card { background: linear-gradient(135deg, #0d1b2a, #1b263b); color: #e0e6ed; padding: 14px 16px; border-radius: 12px; border: 1px solid #243447; box-shadow: 0 4px 12px rgba(0,0,0,0.18); }
    .card h4 { margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.3px; color: #8dc6ff; }
    .card .value { font-size: 1.4rem; font-weight: 700; }
    .card .sub { font-size: 0.8rem; color: #9fb3c8; }
    </style>
    """, unsafe_allow_html=True)

    # Card strip
    st.markdown(
        f"""
        <div class="card-grid">
            <div class="card">
                <h4>Dataset</h4>
                <div class="value">{len(Xp):,}</div>
                <div class="sub">Preprocessed samples</div>
            </div>
            <div class="card">
                <h4>Model Ensemble</h4>
                <div class="value">3</div>
                <div class="sub">Hybrid (wide/deep/balanced)</div>
            </div>
            <div class="card">
                <h4>Overall Accuracy</h4>
                <div class="value">{(metrics.get('accuracy') if metrics_ready else 0):.2f}%</div>
                <div class="sub">Balanced eval set</div>
            </div>
            <div class="card">
                <h4>Critical Errors</h4>
                <div class="value">{(errors.get('1_to_5', 0) if errors else 0)} / {(errors.get('5_to_1', 0) if errors else 0)}</div>
                <div class="sub">1->5 / 5->1</div>
            </div>
            <div class="card">
                <h4>Eval Samples</h4>
                <div class="value">{(metrics.get('total_samples') if metrics_ready else 0):,}</div>
                <div class="sub">Balanced per class</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top:16px; padding:12px 16px; background: #0f1926; border:1px solid #243447; border-radius:12px;">
          <h4 style="margin:0 0 8px 0; color:#8dc6ff;">What you can do</h4>
          <ul style="margin:0; padding-left:18px; color:#dbe7f0; line-height:1.5;">
            <li><strong>Single Prediction:</strong> choose a sample index to see predicted CSAT, confidence, and class probabilities.</li>
            <li><strong>Batch Testing:</strong> run a set of indices (random or manual), get accuracy, critical error counts, and per-row results.</li>
            <li><strong>Model Performance:</strong> view cached accuracy, confusion matrix, critical errors, and per-class accuracy on the balanced eval set.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif menu_choice == "Single Prediction":
    st.header("Single CSAT Prediction")
    if max_index < 0:
        st.error("No data available.")
        st.stop()
    col1, col2 = st.columns([2, 1])
    with col1:
        num_val = st.number_input(
            "Sample Index",
            min_value=0,
            max_value=max_index,
            value=int(st.session_state.sample_index),
            step=1,
            key="sample_index_input",
        )
        if int(num_val) != st.session_state.sample_index:
            st.session_state.sample_index = int(num_val)
    with col2:
        if st.button("Get Random Sample"):
            st.session_state.sample_index = int(st.session_state.rng.integers(0, max_index + 1))
            st.experimental_rerun()
    idx = int(st.session_state.sample_index)
    st.info(f"Using sample {idx} of {max_index}")
    sample_features = Xp[idx:idx+1]
    actual_csat = int(y_full.iloc[idx])
    preds, probs = hybrid_predict(models, sample_features)
    probs = sanitize_probabilities(probs)
    pred_csat = int(preds[0])
    prob_vals = np.squeeze(sanitize_probabilities(probs[0]))
    confidence = float(np.max(prob_vals) * 100.0)
    st.success(f"Predicted: {pred_csat}/5 | Actual: {actual_csat}/5 | Confidence: {confidence:.1f}%")
    if pred_csat != actual_csat:
        st.warning(f"Misclassified: predicted CSAT {pred_csat} vs actual {actual_csat}")
    else:
        st.info("Classification: correct")
    st.caption(f"Probabilities (sum={prob_vals.sum():.3f}): {np.round(prob_vals,4).tolist()}")
    st.subheader("Probability Breakdown")
    prob_df = pd.DataFrame({"CSAT": [f"CSAT {i+1}" for i in range(5)], "Probability": prob_vals.tolist()})
    st.dataframe(prob_df, use_container_width=True)

elif menu_choice == "Batch Testing":
    st.header("Batch CSAT Testing")
    if max_index < 0:
        st.error("No data available.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.selectbox("Test Size", [10, 25, 50, 100], index=1)
        if st.button("Test Random Samples"):
            size = min(test_size, max_index + 1)
            st.session_state.batch_indices = [int(i) for i in st.session_state.rng.choice(max_index + 1, size=size, replace=False)]
            st.experimental_rerun()
    with col2:
        manual_indices = st.text_input("Manual Indices", placeholder="1,5,10")
        if st.button("Test Manual Indices"):
            try:
                st.session_state.batch_indices = [int(x.strip()) for x in manual_indices.split(',') if x.strip()]
                st.experimental_rerun()
            except Exception:
                st.error("Provide comma-separated integers.")

    indices = st.session_state.get('batch_indices', [])
    if indices:
        valid = [i for i in indices if 0 <= i <= max_index]
        if not valid:
            st.error("No valid indices.")
        else:
            X_batch = Xp[valid]
            y_batch = y_full.iloc[valid].to_numpy()
            preds, probs = hybrid_predict(models, X_batch)
            probs = sanitize_probabilities(probs)
            correct = (preds == y_batch)
            accuracy = float(np.mean(correct) * 100.0)
            error_counts = {
                '1_to_5': int(np.sum((y_batch == 1) & (preds == 5))),
                '5_to_1': int(np.sum((y_batch == 5) & (preds == 1))),
                'other': int(np.sum((preds != y_batch) & ~(((y_batch == 1) & (preds == 5)) | ((y_batch == 5) & (preds == 1))))),
            }
            st.success(f"Batch testing completed. Accuracy: {accuracy:.1f}% on {len(valid)} samples.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tested", len(valid))
            c2.metric("Accuracy", f"{accuracy:.1f}%")
            c3.metric("Correct", int(correct.sum()))
            c4.metric("Critical Errors", error_counts['1_to_5'] + error_counts['5_to_1'])

            st.subheader("Error Analysis")
            error_df = pd.DataFrame([
                {"Error Type": "1->5 (Missed Dissatisfied)", "Count": error_counts['1_to_5'], "Impact": "High"},
                {"Error Type": "5->1 (False Alarm)", "Count": error_counts['5_to_1'], "Impact": "Medium"},
                {"Error Type": "Other Misclassification", "Count": error_counts['other'], "Impact": "Low"},
            ])
            st.dataframe(error_df, use_container_width=True)

            st.subheader("Batch Results")
            res_df = pd.DataFrame({
                "Index": valid,
                "Predicted": preds,
                "Actual": y_batch,
                "Confidence": np.max(probs, axis=1) * 100.0,
                "Probabilities": [sanitize_probabilities(p)[0].tolist() for p in probs],
                "Correct": correct,
            })
            st.dataframe(res_df, use_container_width=True)

elif menu_choice == "Model Performance":
    st.header("Model Performance Analytics")
    metrics = compute_metrics_cached()
    if not metrics or not metrics.get('success'):
        err_msg = metrics.get('error') if isinstance(metrics, dict) else 'unknown error'
        st.error(f"Metrics unavailable: {err_msg}")
        st.stop()
    else:
        st.caption(f"Metrics debug: accuracy={metrics.get('accuracy')}, samples={metrics.get('total_samples')}")
        errors = metrics['error_breakdown']
        conf = np.array(metrics['confusion_matrix'], dtype=np.int64)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Accuracy", f"{metrics['accuracy']:.2f}%")
        c2.metric("1->5 Errors", errors.get('1_to_5', 0))
        c3.metric("5->1 Errors", errors.get('5_to_1', 0))
        c4.metric("Total Errors", sum(int(v) for v in errors.values()))

        st.subheader("Confusion Matrix")
        labels = [f"CSAT {i}" for i in range(1, 6)]
        st.dataframe(pd.DataFrame(conf, index=labels, columns=labels), use_container_width=True)

        st.subheader("Critical Error Breakdown")
        critical_df = pd.DataFrame([
            {"Error Type": "1->5 (Missed Dissatisfied)", "Count": errors.get('1_to_5', 0)},
            {"Error Type": "5->1 (False Alarm)", "Count": errors.get('5_to_1', 0)},
            {"Error Type": "Other Misclassification", "Count": errors.get('other', 0)},
        ])
        st.dataframe(critical_df, use_container_width=True)

        per_class = metrics.get('per_class_accuracy', [])
        if per_class:
            st.subheader("Per-Class Accuracy")
            st.dataframe(pd.DataFrame(per_class), use_container_width=True)

elif menu_choice == "About":
    st.header("About")
    st.write(
        """
        This Streamlit app runs the CSAT hybrid ensemble locally. Models and preprocessing are loaded from disk; predictions and metrics are computed in-process without any API.
        """
    )

    st.subheader("Model Comparison (from latest training run)")
    static_metrics = load_static_metrics()
    if static_metrics:
        rows = []
        for name, vals in static_metrics.items():
            if name == "timestamp" or str(name).startswith("__"):
                continue
            errs = vals.get("errors", {}) if isinstance(vals, dict) else {}
            acc = vals.get("accuracy") if isinstance(vals, dict) else None
            rows.append({
                "Model": name.replace("_", " ").title(),
                "Accuracy": f"{acc:.2f}%" if isinstance(acc, (int, float)) else "N/A",
                "1->5": errs.get("1_to_5", "N/A"),
                "5->1": errs.get("5_to_1", "N/A"),
                "Total 1-5": errs.get("total_1to5", "N/A"),
            })
        df_rows = pd.DataFrame(rows).astype(str)
        st.table(df_rows)
        ts = static_metrics.get("timestamp")
        if ts:
            st.caption(f"Metrics timestamp: {ts}")
        src = static_metrics.get("__source")
        if src:
            st.caption(f"Loaded from: {src}")
    else:
        st.info("No model_metrics.json found. Run the notebook to export metrics.")