from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from sklearn.datasets import load_iris


st.set_page_config(page_title="Iris ??璅∪??葫", page_icon="?", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(120deg, #f8fafc 0%, #e0f2fe 100%);
    }
    .main-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        box-shadow: 0 10px 25px rgba(14, 116, 144, 0.08);
    }
    .result-card {
        background: #ecfeff;
        border: 1px solid #a5f3fc;
        border-radius: 14px;
        padding: 1rem;
    }
    .model-tag {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: #cffafe;
        color: #0f766e;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_FILES = {
    "KNN": "knn_model.joblib",
    "蝢??航艘甇?: "logistic_model.joblib",
    "擃鞎???: "naive_bayes_model.joblib",
    "XGBoost": "xgb_model.joblib",
}

TARGET_NAMES = load_iris().target_names
iris = load_iris()
feature_names = iris.feature_names


@st.cache_resource
def load_model(model_filename: str):
    return joblib.load(model_filename)


@st.cache_resource
def load_scaler():
    scaler_iris_path = Path("scaler_iris.joblib")
    if scaler_iris_path.exists():
        return joblib.load(scaler_iris_path), scaler_iris_path.name
    fallback = Path("scaler.joblib")
    if fallback.exists():
        return joblib.load(fallback), fallback.name
    raise FileNotFoundError("?曆???scaler_iris.joblib ??scaler.joblib")


st.title("Iris ??璅∪??葫 APP")
st.caption("?豢?璅∪?銝西撓?亦敺萄潘?????蝡?葫 Iris ?車??)

with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)

    selected_model_name = st.selectbox("?豢???璅∪?", list(MODEL_FILES.keys()))
    st.markdown(
        f"<span class='model-tag'>?桀?璅∪?嚗selected_model_name}</span>",
        unsafe_allow_html=True,
    )

    input_values = []
    for idx, feature in enumerate(feature_names):
        col = iris.data[:, idx]
        input_values.append(
            st.slider(
                label=feature,
                min_value=float(np.min(col)),
                max_value=float(np.max(col)),
                value=float(np.median(col)),
                step=0.1,
            )
        )

    predict_btn = st.button("???葫", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    try:
        model = load_model(MODEL_FILES[selected_model_name])
        scaler, scaler_name = load_scaler()

        raw_input = np.array([input_values], dtype=float)
        scaled_input = scaler.transform(raw_input)
        pred_idx = int(model.predict(scaled_input)[0])

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("?葫蝯?")
        st.write(f"?葫?車嚗?*{TARGET_NAMES[pred_idx]}**")
        st.write(f"雿輻璅??嚗{scaler_name}`")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_input)[0]
            st.write("???交???")
            for class_name, prob in zip(TARGET_NAMES, probs):
                st.write(f"- {class_name}: {prob:.2%}")

        st.caption("撌脣???皞?敺??脰?璅∪??葫??)
        st.markdown("</div>", unsafe_allow_html=True)
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"?葫憭望?嚗error}")
