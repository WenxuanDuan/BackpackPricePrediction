import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils import preprocess_user_input

st.set_page_config(page_title="Backpack Price Predictor", layout="centered")

# ------------------------- 🎯 LOAD MODELS AND REFERENCES -------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/stacking_model.pkl")

@st.cache_resource
def load_preprocessing_files():
    ref_cols = joblib.load("models/reference_columns.pkl")
    weight_max = joblib.load("models/weight_max.pkl")
    scaler = joblib.load("models/minmax_scaler.pkl")
    scaler_cols = joblib.load("models/scaler_columns.pkl")
    return ref_cols, weight_max, scaler, scaler_cols

# 加载模型和处理器
model = load_model()
reference_columns, weight_max, minmax_scaler, scaler_columns = load_preprocessing_files()


# ------------------------- 🎨 PAGE UI -------------------------
st.title("🎒 Backpack Price Predictor")
st.markdown("Predict the price💰of your backpack based on key features you choose.")
st.markdown("---")
st.header("🧾 Select Backpack Features")

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

compartments = st.slider("Compartments", 1, 10, 3)
weight = st.slider("Weight Capacity (kg)", 5.0, 30.0, 10.0, step=0.5)

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

st.markdown("---")

# 在 predict 按钮中处理：
if st.button("🎯 Predict Price"):

    with st.spinner("Predicting price... Please wait ⏳"):
        try:
            input_df = preprocess_user_input(user_input, reference_columns, weight_max, minmax_scaler, scaler_columns)

            # 调试：确认列一致性
            missing = set(reference_columns) - set(input_df.columns)
            extra = set(input_df.columns) - set(reference_columns)
            if missing:
                st.error(f"Missing columns: {missing}")
            if extra:
                st.warning(f"Extra columns not needed: {extra}")

            pred_price = model.predict(input_df)[0]
            # st.write("🎯 Input DF columns:", sorted(input_df.columns.tolist()))
            # st.write("🎯 Reference columns:", sorted(reference_columns))
            st.success(f"💸 Predicted Backpack Price: ${pred_price:.2f}")
            # st.write("🧪 Transformed input:", input_df)

        except Exception as e:
            st.error("❌ An error occurred during prediction.")
            st.exception(e)

# Optional footer
st.markdown("---")
st.markdown("<div style='text-align:center; font-size:13px;'>Made by Wenxuan Duan</div>", unsafe_allow_html=True)

