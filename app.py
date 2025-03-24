import streamlit as st
st.set_page_config(page_title="Backpack Price Predictor", layout="centered")

import joblib
import pandas as pd
from codes.dataPreprocessing import preprocess_data
from utils import preprocess_user_input

# ------------------------- 🎯 CACHE MODEL -------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/stacking_model.pkl")

# 加载模型
model = load_model()

# 加载参考列（特征对齐使用）
train_X, _, _, _ = preprocess_data("dataset/train.csv", "dataset/test.csv")
reference_columns = train_X.columns.tolist()
weight_max = 30.0  # 从训练集中最大值提取（也可动态设置）

# ------------------------- 🎨 PAGE UI -------------------------


st.title("🎒 Backpack Price Predictor")
st.markdown("Predict the price💰of your backpack based on key features you choose.")

st.markdown("---")
st.header("🧾 Select Backpack Features")

# ------------------------- USER INPUT -------------------------
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

# ------------------------- PREDICT -------------------------
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
if st.button("🎯 Predict Price"):
    with st.spinner("Predicting price... Please wait ⏳"):
        try:
            input_df = preprocess_user_input(user_input, reference_columns, weight_max)
            pred_price = model.predict(input_df)[0]
            st.success(f"💸 Predicted Backpack Price: **${pred_price:.2f}**")
        except Exception as e:
            st.error("❌ An error occurred during prediction.")
            st.exception(e)
