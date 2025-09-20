# Customer Churn Prediction

## 🎯 Project Goal
The goal of this project is to build a machine learning model that can predict whether a customer is likely to churn (leave a service) based on their historical data. This helps businesses proactively identify at-risk customers.

---
## 🚀 Interactive Web App (NEW SECTION)
This project includes an interactive web application built with Streamlit. You can upload your own dataset (in the same format as the Telco dataset) and the app will automatically train the models and display the performance results.

### How to Run the App
1.  Make sure you have Python and the required libraries installed:
    ```bash
    pip install streamlit pandas scikit-learn imblearn matplotlib seaborn
    ```
2.  Clone or download this repository to your local machine.
3.  Navigate to the project folder in your terminal and run the following command:
    ```bash
    streamlit run app.py
    ```
---

## 📖 Methodology
I followed these steps to build the prediction models:
1.  **Data Loading & Cleaning:** Loaded the Telco Customer Churn dataset and cleaned it by converting data types and handling missing values.
2.  **Exploratory Data Analysis (EDA):** Analyzed the data to understand the features and confirmed the class imbalance in the churn variable.
3.  **Data Preprocessing:** Encoded categorical features into numerical format and scaled the data using `StandardScaler`.
4.  **Handling Imbalance:** Used the **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data to create a balanced dataset for the models to learn from.
5.  **Modeling:** Trained and evaluated three different classification models:
    * Logistic Regression
    * Decision Tree
    * Simple Neural Network (MLPClassifier)
6.  **Evaluation:** Assessed the models based on precision, recall, and F1-score, focusing on the recall for the "Churn" class as the key performance metric.

---

## 📈 Results
The models were evaluated on an unseen test set. The **Logistic Regression** model performed the best, achieving a **recall of 63%** for the churn class. This means it was the most effective model at correctly identifying the customers who were actually going to churn.

| Model                 | Recall (for Churn) |
| --------------------- | ------------------ |
| **Logistic Regression** | **0.63** |
| Neural Network        | 0.53               |
| Decision Tree         | 0.53               |

---

## 💻 Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Imblearn (for SMOTE)
* Jupyter Notebook
