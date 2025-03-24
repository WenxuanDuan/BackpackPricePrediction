# 🎒 Backpack Price Predictor

A machine learning web application that predicts the price of a backpack based on selected features like brand, material, size, and more.


---

## 🚀 Live Demo

👉 Try it here: https://backpackpriceprediction-wxd.streamlit.app/
*(Replace this with your actual Streamlit Cloud link)*

---

## 📌 Features

- 🧼 Automated data preprocessing: missing values, outliers, feature engineering
- 🧠 Model training & evaluation with 10-fold cross-validation
- ⚙️ Stacking ensemble model with Ridge, LightGBM, and MLP
- 📊 Performance visualizations: RMSE, prediction errors, true vs predicted prices
- 🌐 Interactive Streamlit Web App
- 📦 Exportable predictions


---

## ⚙️ How It Works

### 1. Data Preprocessing
- Missing value imputation
- Feature engineering: interaction terms, binning, normalization
- One-hot encoding for categorical features
- Outlier removal (IQR)
- Log-transform for skewed numerical features (e.g. weight capacity)

### 2. Model Training
- Multiple regressors evaluated (Random Forest, LightGBM, CatBoost, etc.)
- Final model: **StackingRegressor**  
  - Base: Ridge, LightGBM, MLP  
  - Meta: ElasticNet

### 3. Web Application
- Users select backpack features (brand, size, etc.)
- App processes inputs and predicts the price using trained model
- Built with **Streamlit** for easy sharing and demo

---

## 📈 Results

| Model                                           | RMSE (↓)  | Time per Fold (sec) |
|-------------------------------------------------|-----------|---------------------|
| Random Forest                                   | 40.38     | 47.10               |
| ExtraTrees                                      | 43.63     | 23.50               |
| KNN (k=5)                                       | 42.71     | 3.91                |
| Ridge                                           | 39.03     | 0.13                |
| Lasso                                           | 39.03     | 0.28                |
| ElasticNet                                      | 39.03     | 0.32                |
| LightGBM                                        | 39.03     | 0.92                |
| CatBoost                                        | 39.12     | 0.73                |
| XGBoost                                         | 39.21     | 0.68                |
| HistGradientBoosting                            | 39.03     | 1.06                |
| MLP                                             | 39.07     | 16.56               |
| Voting (Ridge+LGBM+MLP)                         | 39.03     | 19.30               |
| Stacking (Ridge+LGBM+MLP, ElasticNet)           | **39.02** | 58.66               |
| Stacking (Ridge+LGBM+MLP, HistGradientBoosting) | 39.03     | 58.80               |

