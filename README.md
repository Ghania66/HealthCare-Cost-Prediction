# 🏥 Healthcare Cost Prediction using Regression

This project predicts **individual healthcare expenses** based on demographic and lifestyle data using a **regression-based deep learning model** built with TensorFlow and Keras.

---

## 📘 Project Overview

Healthcare costs vary widely depending on personal factors like age, BMI, smoking habits, and region.  
This project applies **machine learning regression** to predict medical insurance costs accurately within **$3500 mean absolute error (MAE)** using the [Insurance Dataset](https://www.kaggle.com/mirichoi0218/insurance).

---

## 🧠 Objective

Develop and evaluate a **neural network regression model** capable of predicting healthcare costs with high accuracy.

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Splitting & Scaling | scikit-learn |
| Machine Learning | TensorFlow, Keras |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook / Google Colab |

---

## 🧩 Steps Performed

1. **Data Loading**  
   - Loaded `insurance.csv` containing age, sex, BMI, children, smoker, region, and expenses.

2. **Data Preprocessing**  
   - Converted categorical variables (`sex`, `smoker`, `region`) into numeric using one-hot encoding.  
   - Split dataset into **80% training** and **20% testing** sets.  
   - Scaled input features using `StandardScaler`.  
   - Normalized target labels (`expenses / 1000`).

3. **Model Development**  
   - Built a regression model using **Keras Sequential API**:
     ```python
     model = keras.Sequential([
         keras.Input(shape=(train_dataset.shape[1],)),
         layers.Dense(256, activation='relu'),
         layers.Dense(128, activation='relu'),
         layers.Dense(64, activation='relu'),
         layers.Dense(1)
     ])
     ```
   - Optimizer: `Adam(learning_rate=0.001)`  
   - Loss: `Mean Squared Error (MSE)`  
   - Metrics: `MAE`, `MSE`

4. **Training & Evaluation**  
   - Trained the model for **300 epochs** with validation split (20%).  
   - Evaluated on test data using unseen samples.

5. **Performance Metric**  
   - Achieved **Mean Absolute Error < 3500**, meeting project requirement.  
   - Visualized true vs. predicted healthcare costs using scatter plots.

---

## 📊 Results

| Metric | Value |
|--------|--------|
| Mean Absolute Error (MAE) | ≈ 2500–3200 |
| Loss Function | MSE |
| Optimizer | Adam |


