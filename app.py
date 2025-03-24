import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils import preprocess_user_input

# ------------------------- 🔧 Streamlit Config -------------------------
st.set_page_config(page_title="Backpack Price Predictor", layout="centered")

# ------------------------- 📦 Caching -------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/stacking_model.pkl")

@st.cache_resource
def load_reference_columns():
    return joblib.load("models/reference_columns.pkl")

@st.cache_resource
def load_weight_max():
    return joblib.load("models/weight_max.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("models/minmax_scaler.pkl")

# ------------------------- 🔄 Load Model & Resources -------------------------

model = load_model()
reference_columns = load_reference_columns()
weight_max = load_weight_max()
scaler = load_scaler()

# ------------------------- 🎨 Page UI -------------------------

st.title("🎒 Backpack Price Predictor")
st.markdown("Predict the price 💰 of your backpack based on key features.")

st.markdown("---")
st.header("🧾 Select Backpack Features")

# ------------------------- 🧍 User Inputs -------------------------

brand = st.selectbox("Brand", ['Adidas', 'Jansport', 'Nike', 'Puma', 'Under Armour'])
material = st.selectbox("Material", ['Canvas', 'Leather', 'Nylon', 'Polyester'])
size = st.selectbox("Size", ['Large', 'Medium', 'Small'])
style = st.selectbox("Style", ['Backpack', 'Messenger', 'Tote'])
color = st.selectbox("Color", ['Black', 'Blue', 'Gray', 'Green', 'Pink', 'Red'])

col1, col2 = st.columns(2)
with col1:
    laptop_compartment = st.radio("Laptop Compartment", ['Yes', 'No'])
with col2:
    waterproof = st.radio("Waterproof", ['Yes', 'No'])

compartments = st.slider("Compartments", min_value=1, max_value=10, value=3)
weight = st.slider("Weight Capacity (kg)", min_value=5.0, max_value=30.0, value=10.0, step=0.5)

# ------------------------- 📋 Pack User Inputs -------------------------

user_input = {
    "Brand": brand,
    "Material": material,
    "Size": size,
    "Style": style,
    "Color": color,
    "Laptop Compartment": laptop_compartment,
    "Waterproof": waterproof,
    "Compartments": compartments,
    "Weight Capacity (kg)": weight
}

# ------------------------- 🎯 Predict -------------------------

st.markdown("---")
if st.button("🔮 Predict Backpack Price"):
    with st.spinner("Predicting... please wait ⏳"):
        try:
            input_df = preprocess_user_input(user_input, reference_columns, weight_max, scaler)
            price = model.predict(input_df)[0]
            st.success(f"💸 Predicted Backpack Price: ${price:.2f}")

        except Exception as e:
            st.error("❌ Something went wrong during prediction.")
            st.exception(e)

# Optional footer
st.markdown("---")
st.markdown("<div style='text-align:center; font-size:13px;'>Made with ❤️ by Team Backpack</div>", unsafe_allow_html=True)
