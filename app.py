import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Set the page configuration for the app
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# --- Main App Logic ---
def main():
    """The main function that runs the Streamlit app."""
    
    st.title("🚀 Customer Churn Prediction App")
    st.write("Upload your customer dataset to train models and see their performance.")

    # --- Sidebar for File Upload ---
    with st.sidebar:
        st.header("1. Upload Your Data")
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    # --- Main Panel Logic ---
    if uploaded_file is not None:
        try:
            # Load the uploaded data
            df = pd.read_csv(uploaded_file)
            
            st.success("File uploaded successfully! Here's a preview of your data:")
            st.dataframe(df.head())

            # --- Data Preprocessing ---
            st.header("⚙️ Data Preprocessing & Model Training")
            with st.spinner("Processing data and training models... This may take a moment."):
                
                # Make a copy to avoid modifying the original dataframe
                df_processed = df.copy()

                # 1. Handle 'TotalCharges' - Convert to numeric and fill missing values
                df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
                df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)

                # 2. Drop customerID
                if 'customerID' in df_processed.columns:
                    df_processed.drop('customerID', axis=1, inplace=True)

                # 3. Encode categorical features and the target variable
                categorical_cols = df_processed.select_dtypes(include=['object']).columns.drop('Churn')
                df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
                
                le = LabelEncoder()
                df_processed['Churn'] = le.fit_transform(df_processed['Churn'])

                # 4. Separate features and target
                X = df_processed.drop('Churn', axis=1)
                y = df_processed['Churn']

                # 5. Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # 6. Apply SMOTE to the training data
                smote = SMOTE(random_state=42)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

                # 7. Scale numerical features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_smote)
                X_test_scaled = scaler.transform(X_test)

                # --- Model Training ---
                models = {
                    "Logistic Regression": LogisticRegression(random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Neural Network": MLPClassifier(random_state=42, max_iter=500)
                }

                results = {}
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train_smote)
                    y_pred = model.predict(X_test_scaled)
                    results[name] = {
                        "report": classification_report(y_test, y_pred, output_dict=True),
                        "cm": confusion_matrix(y_test, y_pred)
                    }

            st.success("Models trained successfully!")
            
            # --- Display Results ---
            st.header("📊 Model Performance Evaluation")
            
            for name, result in results.items():
                st.subheader(f"Results for {name}")
                
                # Display classification report as a dataframe for better visualization
                report_df = pd.DataFrame(result['report']).transpose()
                st.table(report_df.round(2))
                
                # Display confusion matrix
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.heatmap(result['cm'], annot=True, fmt='g', ax=ax, cmap='Blues')
                ax.set_title(f'{name} Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                st.markdown("---")


        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please ensure your CSV file has the same structure as the Telco dataset, including a 'Churn' column.")

    else:
        st.info("Awaiting for CSV file to be uploaded.")

if __name__ == '__main__':
    main()
