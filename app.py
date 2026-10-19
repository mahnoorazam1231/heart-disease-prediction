
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details.")

@st.cache_resource
def load_model():
    with open(Path(__file__).parent / "heart_model.pkl", "rb") as f:
        return pickle.load(f)

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4])
thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)

if st.button("Predict"):
    model = load_model()
    data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )
    result = model.predict(data)[0]

    if result == 0:
        st.success("Lower likelihood of heart disease.")
    else:
        st.warning("Possible heart disease detected.")

    st.caption("Educational prediction only, not a medical diagnosis.")
