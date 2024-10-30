import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Bank Account Prediction", layout="wide")
st.title("Bank Account Prediction")

# Load the model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
        st.write(f"Loaded model type: {type(model)}")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Subheader for the input form
st.subheader("Enter the following features to predict Bank account:")

# Collect user inputs for the form
country = st.selectbox("Country", ["Kenya", "Uganda", "Tanzania", "Rwanda", "Other"])
location_type = st.radio("Location Type", ["Rural", "Urban"])
cellphone_access = st.selectbox("Cellphone Accessibility", ["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=1)
age = st.number_input("Age of Respondent", min_value=18, max_value=100, value=25)
gender = st.radio("Gender of Respondent", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Married/Living together", "Single/Never Married", "Divorced/Separated", "Widowed"])
education_level = st.selectbox("Education Level", ["No formal education", "Primary education", "Secondary education", "Tertiary education"])
job_type = st.selectbox("Job Type", ["Self employed", "Government Dependent", "Formally employed Private", "Formally employed Government", "Other", "Don't Know/Refuse to answer"])
relationship_with_head = st.selectbox("Relationship with Head", ["Head of Household", "Spouse", "Child", "Parent", "Other"])

# Create a DataFrame for the input
user_input = pd.DataFrame({
    'country': [country],
    'location_type': [location_type],
    'cellphone_access': [cellphone_access],
    'household_size': [household_size],
    'age_of_respondent': [age],
    'gender_of_respondent': [gender],
    'marital_status': [marital_status],
    'education_level': [education_level],
    'job_type': [job_type],
    'relationship_with_head': [relationship_with_head]
})

st.write(f"Input data shape: {user_input.shape}")

# Ensure model is correctly loaded and process prediction
if st.button("Predict"):
    try:
        if hasattr(model, "predict"):
            # Prepare input for prediction
            user_input_encoded = pd.get_dummies(user_input)
            user_input_encoded = user_input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
            
            prediction = model.predict(user_input_encoded)[0]
            st.write(f"Prediction: {'Has Bank Account' if prediction == 1 else 'No Bank Account'}")
        else:
            st.error("The loaded model does not appear to be valid for prediction.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
