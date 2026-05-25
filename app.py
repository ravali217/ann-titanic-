import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# PAGE CONFIG
# ------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------
# LOAD MODEL
# ------------------------------------

model = tf.keras.models.load_model("titanic_ann_model.h5")

# ------------------------------------
# HEADER SECTION
# ------------------------------------

st.markdown("""
# 🚢 Titanic Survival Prediction System
### Deep Learning Based Passenger Survival Prediction
""")

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
    use_container_width=True
)

# ------------------------------------
# PROJECT DESCRIPTION
# ------------------------------------

st.markdown("""
## 📌 Project Description

This application predicts whether a passenger would survive during the Titanic disaster using:

- Artificial Neural Networks (ANN)
- TensorFlow Deep Learning Model
- Streamlit Deployment

The model was trained using passenger details such as:
- Passenger Class
- Age
- Fare

The trained ANN model performs prediction using forward propagation and outputs survival probability.
""")

st.divider()

# ------------------------------------
# INPUT SECTION
# ------------------------------------

st.subheader("🧾 Passenger Details")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=120.0
    )

# ------------------------------------
# NORMALIZATION
# SAME AS TRAINING
# ------------------------------------

# Min-Max Scaling

pclass_norm = (pclass - 1) / (3 - 1)
age_norm = (age - 1) / (80 - 1)
fare_norm = (fare - 0) / (600 - 0)

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

# ------------------------------------
# PREDICTION BUTTON
# ------------------------------------

if st.button("🔍 Predict Survival"):

    prediction = model.predict(input_data)

    probability = float(prediction[0][0])

    # --------------------------------
    # PREDICTION LOGIC
    # --------------------------------

    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    confidence = probability * 100

    st.divider()

    # --------------------------------
    # OUTPUT AREA
    # --------------------------------

    st.subheader("📊 Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="Prediction",
            value=result
        )

    with c2:
        st.metric(
            label="Survival Probability",
            value=f"{probability:.4f}"
        )

    with c3:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    # --------------------------------
    # VISUALIZATION
    # --------------------------------

    st.subheader("📈 Probability Visualization")

    survive_prob = probability
    nonsurvive_prob = 1 - probability

    chart_data = pd.DataFrame({
        "Category": ["Survived", "Not Survived"],
        "Probability": [survive_prob, nonsurvive_prob]
    })

    fig, ax = plt.subplots()

    ax.bar(
        chart_data["Category"],
        chart_data["Probability"]
    )

    ax.set_ylabel("Probability")
    ax.set_title("Survival Probability")

    st.pyplot(fig)

    # --------------------------------
    # PIE CHART
    # --------------------------------

    fig2, ax2 = plt.subplots()

    ax2.pie(
        [survive_prob, nonsurvive_prob],
        labels=["Survived", "Not Survived"],
        autopct='%1.1f%%'
    )

    ax2.set_title("Prediction Distribution")

    st.pyplot(fig2)
