import streamlit as st
import pandas as pd
from joblib import load
import pickle
# Function to load model and data
def load_model():
    # Assuming "pipeline.pkl" is in the same directory as your script:
    with open("pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

# Function to show prediction result
def show_prediction(pipeline, sample):
    prediction = pipeline.predict(sample)[0]
    if prediction == 1:
        st.markdown(
            """
            <div style="background-color:#FFEBE6; padding:20px; text-align:center;">
                <h3 style="color:#FF4500;">Based on the input, there's a chance the employee may leave the organization.</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#E6F7E7; padding:20px; text-align:center;">
                <h3 style="color:#008000;">Based on the input, it's likely the employee will stay with the organization.</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

# Streamlit app layout
st.set_page_config(
    page_title="Employee Churn Prediction",
    page_icon="",
    layout="wide"
)

st.title("Predicting Employee Churn Using Machine Learning")

# Load model (ensure correct path)
pipeline = load_model()

# Sidebar to select test type
test_type = st.sidebar.selectbox("Select Test Type:", ["Single Prediction", "Data File Prediction"])

# Single prediction interface
if test_type == "Single Prediction":
    st.subheader("Single Employee Prediction")

    # Employee data input fields
    e1 = st.slider("Employee satisfaction level", 0.0, 1.0, 0.5)
    e2 = st.slider("Last evaluation score", 0.0, 1.0, 0.5)
    e3 = st.slider("Number of projects assigned to", 1, 10, 5)
    e4 = st.slider("Average monthly hours worked", 50, 300, 150)
    e5 = st.slider("Time spent at the company", 1, 10, 3)
    e6 = st.radio("Whether they have had a work accident", [0, 1])
    e7 = st.radio("Whether they have had a promotion in the last 5 years", [0, 1])

    options = ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing',
               'RandD', 'accounting', 'hr', 'management']
    e8 = st.selectbox("Department name", options)

    options1 = ['low', 'medium', 'high']
    e9 = st.selectbox("Salary category", options1)

    # Predict button with clear action
    if st.button("Predict Churn Risk"):
        sample = pd.DataFrame({
            'satisfaction_level': [e1],
            'last_evaluation': [e2],
            'number_project': [e3],
            'average_montly_hours': [e4],
            'time_spend_company': [e5],
            'Work_accident': [e6],
            'promotion_last_5years': [e7],
            'departments': [e8],
            'salary': [e9]
        })
        show_prediction(pipeline, sample)

# Data file prediction interface
else:
    st.subheader("Data File Prediction")

    # File upload for CSV data
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load data from CSV, handle potential issues
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.replace('\n', '')
            data.rename(columns={'Departments ': 'departments'}, inplace=True)
            data = data.drop_duplicates()  # Consider handling duplicates

            # Process data if needed (e.g., feature engineering)
            # ...

            # Predict churn risk for each row
            predictions = pipeline.predict(data)

            # Add prediction to the data
            data['predicted_churn'] = predictions

            # Display or save the processed data with predictions
            if st.button("View Processed Data"):
                st.write(data)
            if st.button("Download Processed Data"):
                data.to_csv('processed_data.csv', index=False)

        except Exception as e:
            st.error(f"Failed to process data: {e}")
