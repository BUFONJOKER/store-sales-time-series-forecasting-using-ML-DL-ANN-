# 🏪📈 Store Sales Time Series Forecasting

## 🔍 Project Overview

This project focuses on **Store Sales Prediction** using traditional **Machine Learning regression models** and an **Artificial Neural Network (ANN)**. The goal is to forecast sales accurately by leveraging historical sales data, promotions, holidays, and categorical features such as store number and product family.

The notebook demonstrates a **complete end-to-end ML workflow**, including:

* Data loading
* Exploratory Data Analysis (EDA)
* Feature engineering & preprocessing
* Model training & evaluation
* Performance comparison

---

## 📂 Dataset Information

* **Source:** Google Drive (Parquet format)
* **Type:** Time series sales data
* **Target Variable:** `sales`
* **Key Features:**

  * Store number
  * Product family
  * Date-based features (day, month, year)
  * Promotions
  * Holidays

---

## 🧰 Libraries & Tools Used

### 🐍 Core Python Libraries

* `pandas`, `numpy`
* `matplotlib`, `seaborn`

### 🤖 Machine Learning

* `scikit-learn`

  * Linear Regression
  * KNN Regressor
  * Random Forest Regressor
* `xgboost` (XGBRegressor)

### 🧠 Deep Learning

* `Keras (TensorFlow backend)`

---

## 📊 Exploratory Data Analysis (EDA)

Key analysis performed:

* Dataset shape & structure
* Missing values & duplicates
* Sales distribution
* Store & product family counts
* Correlation analysis
* Removal of zero-sales records

📌 **Outcome:** Cleaned and well-understood dataset ready for modeling.

---

## 🛠️ Data Preprocessing & Feature Engineering

✔ Dropped irrelevant columns (`id`, `date`)
✔ One-hot encoded categorical features:

* Store numbers
* Product families
* Day, month, year

✔ Feature scaling:

* Standardized `onpromotion` using `StandardScaler`

---

## ✂️ Train-Test Split

* Initial split: **80% Train / 20% Test**
* Secondary split for better generalization

This ensures fair evaluation and reduces overfitting risk.

---

## 🤖 Machine Learning Models Implemented

| Model                | Description               |
| -------------------- | ------------------------- |
| 📐 Linear Regression | Baseline regression model |
| 📍 KNN Regressor     | Distance-based regression |
| 🌲 Random Forest     | Ensemble tree-based model |
| 🚀 XGBoost Regressor | Gradient boosting model   |

### 📏 Evaluation Metric

* **Mean Absolute Error (MAE)** used for both training and testing datasets.

---

## 📈 Model Performance Comparison

* Compared **Train MAE vs Test MAE**
* Visualized results using **bar plots**
* Added a **10% error threshold** reference line

📊 This helped identify **overfitting vs underfitting** behavior.

---

## 🧠 Artificial Neural Network (ANN)

### 🏗️ Architecture

* Input layer (113 features)
* Dense layers:

  * 128 neurons (ReLU)
  * 64 neurons (ReLU)
  * 32 neurons (ReLU)
* Output layer:

  * 1 neuron (Linear activation)

### ⚙️ Training Details

* Loss: **Mean Absolute Error (MAE)**
* Optimizer: **Adam**
* Epochs: `5`
* Batch size: `256`

### 📉 Training Visualization

* Training vs Validation loss curves plotted

---

## 🏆 Final Results

* MAE calculated for:

  * Machine Learning models
  * ANN model
* Combined comparison using bar charts

📌 **Insight:** Ensemble models and ANN generally outperform basic regression models.

---


## 📌 Key Learnings

* Importance of feature engineering in time series data
* Performance comparison across ML & DL models
* Practical use of MAE for regression evaluation
* Handling large datasets efficiently

---


⭐ If you find this project helpful, don’t forget to **star the repository**!
