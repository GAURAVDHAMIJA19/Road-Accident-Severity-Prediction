# ============================================================
# 🚗 Road Accident Severity Prediction System | Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🚦 Road Accident Severity Predictor", layout="wide")

# ---------------- SAFE IMAGE FUNCTION ----------------
def st_image_safe(src, caption=None, width=None):
    try:
        st.image(src, caption=caption, width=width)
    except TypeError:
        try:
            if isinstance(src, str) and src.startswith("http"):
                import requests
                resp = requests.get(src)
                img = Image.open(io.BytesIO(resp.content))
                st.image(img, caption=caption, width=width)
        except Exception:
            pass

# ---------------- MODEL LOADING FUNCTION ----------------
@st.cache_resource
def load_pipeline(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None, f"Pipeline file not found: {p.resolve()}"
    try:
        pipe = joblib.load(p)
        return pipe, None
    except Exception as e:
        return None, f"Failed to load pipeline: {e}"

# ---------------- HEADER ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2490/2490393.png", width=72)
st.title("🚗 Road Accident Severity Prediction System")
st.write("An AI-powered system to predict accident severity based on road and driver conditions.")

# Main banner GIF - road alert radar
st_image_safe(
    "https://sentiance.com/wp-content/uploads/2023/05/Car-Crash.png",
    caption="AI-driven Road Safety Alert System",
    width=800
)

# ---------------- SIDEBAR MODEL LOADING ----------------
st.sidebar.header("⚙️ Model / Settings")
default_pipeline_name = "road_accident_pipeline.joblib"
use_file = st.sidebar.radio("Pipeline source", ("Look in app folder", "Upload pipeline file"))

pipeline = None
load_error = None

if use_file == "Look in app folder":
    pipeline, load_error = load_pipeline(default_pipeline_name)
    if pipeline is None:
        st.sidebar.warning(f"Pipeline not found at ./{default_pipeline_name}. You can upload it instead.")
else:
    uploaded = st.sidebar.file_uploader("Upload pipeline (.joblib/.pkl)", type=["joblib", "pkl"])
    if uploaded is not None:
        try:
            pipeline = joblib.load(uploaded)
        except Exception as e:
            st.sidebar.error(f"❌ Error loading pipeline: {e}")

# ---------------- NAVIGATION ----------------
page = st.sidebar.selectbox("📍 Navigate", ["🏠 Home", "📊 Predict", "ℹ️ About"])

# =====================================================
# 🏠 HOME PAGE
# =====================================================
if page == "🏠 Home":
    st.header("Welcome 👋 to the Road Accident Severity Prediction System")
    st.write("""
    This application predicts the **severity of road accidents** using an AI model trained on
    real-world data — considering factors like weather, lighting, driver behavior, and road conditions.
    """)

    # Road safety themed gif
    st_image_safe(
        "https://media.giphy.com/media/XHSmALfzyAotW/giphy.gif",
        caption="Monitoring live traffic for accident risk ⚠️",
        width=800
    )

    st.subheader("📈 Live Alert Dashboard (Demo)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accident Probability (Today)", "27%", "↑ 5% from yesterday")
    with col2:
        st.metric("Traffic Density Risk", "Medium", "↔ Stable")
    with col3:
        st.metric("Weather Risk Index", "High", "↑ Heavy Rain Expected")

    st.info("Navigate to the 📊 *Predict* tab to enter details and check accident severity predictions.")

# =====================================================
# 📊 PREDICTION PAGE
# =====================================================
elif page == "📊 Predict":
    st.header("🧠 Make a Prediction")

    # Prediction radar gif
    st_image_safe(
        "https://tse1.mm.bing.net/th/id/OIP.9FM1pkLQ8OfAUvBuL42cqwHaE4?pid=Api&P=0&h=180",
        caption="Processing road and driver data for AI-based prediction...",
        width=700
    )

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            day_of_week = st.selectbox("Day of Week", ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"])
            road_type = st.selectbox("Road Type", ["single carriageway","dual carriageway","roundabout","one way street","slip road"])
            weather = st.selectbox("Weather Conditions", ["fine","rainy","snow","fog","windy","other"])
            lighting_cond = st.selectbox("Lighting Conditions", ["daylight","darkness - lights lit","darkness - no lighting","other"])
            road_condition = st.selectbox("Road Condition", ["dry","wet","snow","ice","flood","other"])
            vehicle_type = st.selectbox("Vehicle Type Involved", ["car","motorcycle","bus","truck","bicycle","other"])

        with col2:
            num_vehicles = st.slider("Number of Vehicles Involved", 1, 10, 2)
            speed_limit = st.number_input("Speed Limit (km/h)", min_value=0, max_value=200, value=60, step=5)
            driver_age = st.number_input("Driver Age", min_value=16, max_value=100, value=30)
            driver_gender = st.selectbox("Driver Gender", ["male","female","other"])
            alcohol = st.selectbox("Alcohol Involvement", ["no","yes"])
            daynight = st.selectbox("DayNight", ["day","night"])

        submitted = st.form_submit_button("🚦 Predict Severity")

    if submitted:
        input_dict = {
            "Day of Week": day_of_week,
            "Road Type": road_type,
            "Weather Conditions": weather,
            "Lighting Conditions": lighting_cond,
            "Road Condition": road_condition,
            "Vehicle Type Involved": vehicle_type,
            "Number of Vehicles Involved": num_vehicles,
            "Speed Limit (km/h)": speed_limit,
            "Driver Age": driver_age,
            "Driver Gender": driver_gender,
            "Alcohol Involvement": alcohol,
            "DayNight": daynight
        }

        input_df = pd.DataFrame([input_dict])
        st.subheader("🔍 Input Summary")
        st.dataframe(input_df.T)

        if pipeline is None:
            st.error("❌ No model loaded. Please load or upload a pipeline file from the sidebar.")
        else:
            with st.spinner("🚗 Analyzing road and driver conditions..."):
                try:
                    preds = pipeline.predict(input_df)
                    probs = pipeline.predict_proba(input_df)[0]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                else:
                    class_map = {0: "🟢 Minor", 1: "🟠 Serious", 2: "🔴 Fatal"}
                    pred_label = class_map.get(int(preds[0]), int(preds[0]))
                    st.success(f"**Predicted Severity:** {pred_label}")

                    st.markdown("### 🚨 Accident Severity Probability Breakdown")

                    severity_labels = {0: "🟢 Minor", 1: "🟠 Serious", 2: "🔴 Fatal"}
                    prob_texts = [
                        f"**{severity_labels[i]} accident:** {p*100:.2f}% chance"
                        for i, p in enumerate(probs)
                    ]
                    for text in prob_texts:
                        st.write(text)

                    # Visualize
                    max_idx = int(np.argmax(probs))
                    st.progress(int(probs[max_idx] * 100))

                    if max_idx == 0:
                        st.success("✅ Low chance of serious accidents — conditions seem safe.")
                    elif max_idx == 1:
                        st.warning("⚠️ Moderate risk — drive carefully and stay alert.")
                    else:
                        st.error("🚨 High risk of fatal accident — avoid travel if possible!")

# =====================================================
# ℹ️ ABOUT PAGE
# =====================================================
elif page == "ℹ️ About":
    st.header("📘 About This Project")
    st.write("""
    This project is part of a **Road Safety & AI Initiative (2025)**.  
    The goal is to leverage machine learning to assist in predicting the potential severity of accidents
    based on environmental, vehicular, and driver-related conditions.
    
    **Model Used:** Logistic Regression (Multiclass)  
    **Framework:** Streamlit + Scikit-learn  
    **Developed by:** Gaurav & Team 🚀
    """)

    st_image_safe(
        "https://media.giphy.com/media/3o7TKz3EU4ZvO1qBzW/giphy.gif",
        caption="Stay Alert — Save Lives!",
        width=700
    )

    st.markdown("---")
    st.info("© 2025 | Road Accident Severity Prediction | Streamlit App")


