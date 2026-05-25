import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = tf.keras.models.load_model("titanic_ann_model.keras")

# -----------------------------------
# HEADER SECTION
# -----------------------------------

st.title("🚢 Titanic Survival Prediction System")

st.subheader("Deep Learning Based Passenger Survival Prediction")

st.markdown("""
This application predicts whether a passenger would survive or not
using an Artificial Neural Network (ANN) model built with TensorFlow.
""")

# -----------------------------------
# INPUT SECTION
# -----------------------------------

st.header("📋 Passenger Details")

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
        value=120.0
    )

# -----------------------------------
# NORMALIZATION
# -----------------------------------

# Min-Max Scaling

pclass_norm = (pclass - 1) / (3 - 1)
age_norm = age / 100
fare_norm = fare / 500

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("Predict Survival"):

    prediction = model.predict(input_data)

    probability = float(prediction[0][0])

    # -----------------------------------
    # OUTPUT SECTION
    # -----------------------------------

    st.header("📊 Prediction Result")

    if probability > 0.5:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")

    st.metric(
        label="Survival Probability",
        value=f"{probability:.2f}"
    )

    st.metric(
        label="Confidence Score",
        value=f"{probability * 100:.2f}%"
    )

    # -----------------------------------
    # VISUALIZATION
    # -----------------------------------

    survive = probability
    not_survive = 1 - probability

    fig, ax = plt.subplots()

    labels = ["Survive", "Not Survive"]
    values = [survive, not_survive]

    ax.bar(labels, values)

    ax.set_ylabel("Probability")

    ax.set_title("Prediction Probability")

    st.pyplot(fig)

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.markdown("Developed using Streamlit + TensorFlow") 