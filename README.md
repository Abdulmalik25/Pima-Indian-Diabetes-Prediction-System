
# 🩺 Pima Indian Diabetes Prediction System

An end-to-end machine learning system that predicts the likelihood of diabetes in patients based on the Pima Indians Diabetes dataset. This project integrates exploratory data analysis, model training, feature engineering, evaluation, and deployment via an interactive Streamlit web application.

---

## 🎯 Project Goal

To develop and deploy a real-time diabetes prediction system that:
- Accurately identifies diabetic and non-diabetic individuals
- Evaluates multiple classification algorithms
- Presents results through a clean and interactive UI using Streamlit

---

## 📊 Dataset Overview
- **Source**: Pima Indians Diabetes Dataset from the UCI repository
- **Records**: 768 patients
- **Features**: 8 clinical variables
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness 
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- **Target Variable**: `Outcome` (1 = Diabetic, 0 = Non-Diabetic)

---

## 🧠 Project Workflow

### 1. **Data Cleaning**
- Replaced biologically implausible zeroes (e.g., Glucose = 0) with `NaN`
- Imputed missing values with column means

### 2. **Exploratory Data Analysis**
- Visualized distributions using box plots and violin plots
- Identified correlations between features and diabetes outcome

  <img width="792" height="650" alt="image" src="https://github.com/user-attachments/assets/0806e6d7-df4c-43d1-a93c-2925af8776e1" />


### 3. **Outlier Handling**
- Applied IQR method to cap/floor outliers for key features

### 4. **Feature Engineering**
- Scaled features using `StandardScaler`
- Created interaction terms:
  - `Insulin/Glucose` ratio
  - `BMI/Age` ratio
  - `Pregnancy x BMI` product

<img width="703" height="369" alt="image" src="https://github.com/user-attachments/assets/c797bf86-0850-465b-b974-5da64f194551" />


### 5. **Balancing the Data**
- Applied **SMOTE** to balance class distribution before model training

<img width="512" height="353" alt="image" src="https://github.com/user-attachments/assets/4ff76b92-8d1e-4c8c-9cfc-c6ec604aaead" />


### 6. **Model Training**
Trained and evaluated the following models:

| Model                | Accuracy  |
|---------------------|-----------|
| Logistic Regression | 0.7208    |
| KNN                 | 0.6818    |
| Decision Tree       | 0.7143    |
| SVM                 | 0.7338    |
| Naive Bayes         | 0.6818    |
| Random Forest       | 0.7338    |
| XGBoost             | 0.7338    |

<img width="997" height="599" alt="image" src="https://github.com/user-attachments/assets/aa0413b6-c0f2-4d35-bd78-8419a3ef5e5c" />


### 7. **Model Evaluation (AUC Scores)**

| Model                | AUC Score |
|---------------------|-----------|
| Random Forest       | 0.8097    |
| Logistic Regression | 0.8061    |
| SVM                 | 0.8056    |
| XGBoost             | 0.8002    |
| Naive Bayes         | 0.7981    |
| KNN                 | 0.7652    |
| Tuned KNN           | 0.7535    |
| Decision Tree       | 0.6778    |


<img width="996" height="589" alt="image" src="https://github.com/user-attachments/assets/98203429-4202-4589-9ec6-b85edc7fa519" />

---

## 🌐 Deployment with Streamlit

An interactive UI was built using **Streamlit** to allow users to:
- Input patient details (glucose, BMI, insulin, age, etc.)
- Get real-time prediction and confidence score
- Access the model in a clean, user-friendly interface

👉 **[Click here to try the app →](#)** *(replace with actual link once hosted)*

---

## ✅ Conclusion
- Multiple models were evaluated, with Random Forest, SVM, and Logistic Regression performing the best.
- AUC scores confirm strong discrimination ability, especially with ensemble methods.
- The Streamlit interface transforms the model into a practical, accessible tool for healthcare insights.

---

## 📁 Repository Contents
- `diabetes_app.py` – Streamlit app code
- `diabetes_model.pkl` – Trained model file
- `scaler.pkl` – StandardScaler object
- `requirements.txt` – Dependencies for deployment
- `notebooks/` – EDA, training, and tuning notebooks

---

## 🔗 License
This project is licensed for educational and demonstration purposes.
"""
