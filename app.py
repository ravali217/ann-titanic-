import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ---------------------------------
# LOAD TRAINED MODEL
# ---------------------------------

model = tf.keras.models.load_model("titanic_ann_model.h5")

# ---------------------------------
# HEADER SECTION
# ---------------------------------

st.title("🚢 Titanic Survival Prediction System")

st.subheader(
    "Deep Learning Based Passenger Survival Prediction"
)

# ---------------------------------
# PROJECT DESCRIPTION
# ---------------------------------

st.markdown("""
This web application predicts whether a passenger would survive during an emergency situation using an Artificial Neural Network (ANN) model built with TensorFlow and Keras.
""")

st.divider()

# ---------------------------------
# PASSENGER INPUT SECTION
# ---------------------------------

st.header("🧾 Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        min_value=1,
        max_value=80,
        value=24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

# ---------------------------------
# DATA PREPROCESSING
# ---------------------------------

# Normalization
pclass_norm = pclass / 3
age_norm = age / 100
fare_norm = fare / 600

# ---------------------------------
# PREDICTION BUTTON
# ---------------------------------

if st.button("Predict Survival"):

    # Prepare Input
    input_data = np.array([
        [pclass_norm, age_norm, fare_norm]
    ])

    # Model Prediction
    prediction = model.predict(input_data)

    probability = prediction[0][0]

    # Prediction Logic
    if probability > 0.5:
        result = "✅ Passenger Survived"
    else:
        result = "❌ Passenger Not Survived"

    st.divider()

    # ---------------------------------
    # OUTPUT SECTION
    # ---------------------------------

    st.header("📊 Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="Prediction",
            value=result
        )

    with c2:
        st.metric(
            label="Survival Probability",
            value=f"{probability:.2f}"
        )

    with c3:
        st.metric(
            label="Confidence Score",
            value=f"{probability * 100:.2f}%"
        )

    # ---------------------------------
    # VISUALIZATION SECTION
    # ---------------------------------

    st.subheader("📈 Probability Visualization")

    labels = ["Survival", "Non-Survival"]

    values = [
        probability,
        1 - probability
    ]

    fig, ax = plt.subplots()

    ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    st.pyplot(fig)

# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.markdown(
    "Developed using TensorFlow, Keras, and Streamlit"
)