import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pickle
import numpy as np


#Streamlit Code:

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and scaler
with open(r'RandomForestClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open(r'scaling_classify.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=50)
gender = st.selectbox('Select your gender', ('M', 'F'))
smoking = st.selectbox('Do you smoke?', ('Yes', 'No'))
yellow_fingers = st.selectbox('Do you have yellow fingers?', ('Yes', 'No'))
anxiety = st.selectbox('Do you suffer from anxiety?', ('Yes', 'No'))
peer_pressure = st.selectbox('Is there peer pressure?', ('Yes', 'No'))
chronic_disease = st.selectbox('Do you have a chronic disease?', ('Yes', 'No'))
fatigue = st.selectbox('Do you experience fatigue?', ('Yes', 'No'))
allergy = st.selectbox('Do you have any allergy?', ('Yes', 'No'))
wheezing = st.selectbox('Do you have wheezing?', ('Yes', 'No'))
alcohol_consumption = st.selectbox('Do you drink liquor?', ('Yes', 'No'))
coughing = st.selectbox('Do you often cough?', ('Yes', 'No'))
shortness_of_breath = st.selectbox('Do you feel shortness of breath?', ('Yes', 'No'))
swallowing_difficulty = st.selectbox('Do you have trouble swallowing?', ('Yes', 'No'))
chest_pain = st.selectbox('Do you have regular chest pain?', ('Yes', 'No'))

# Map selectbox values
gender_map = {'M': 1, 'F': 0}
smoking_map = {'Yes': 2, 'No': 1}
yes_no_map = {'Yes': 2, 'No': 1}

# Create input data
input_data = pd.DataFrame({
    'GENDER': [gender_map[gender]],
    'AGE': [age],
    'SMOKING': [smoking_map[smoking]],
    'YELLOW_FINGERS': [yes_no_map[yellow_fingers]],
    'ANXIETY': [yes_no_map[anxiety]],
    'PEER_PRESSURE': [yes_no_map[peer_pressure]],
    'CHRONIC DISEASE': [yes_no_map[chronic_disease]],
    'FATIGUE': [yes_no_map[fatigue]],
    'ALLERGY': [yes_no_map[allergy]],
    'WHEEZING': [yes_no_map[wheezing]],
    'ALCOHOL CONSUMING': [yes_no_map[alcohol_consumption]],
    'COUGHING': [yes_no_map[coughing]],
    'SHORTNESS OF BREATH': [yes_no_map[shortness_of_breath]],
    'SWALLOWING DIFFICULTY': [yes_no_map[swallowing_difficulty]],
    'CHEST PAIN': [yes_no_map[chest_pain]]
})

# Make prediction
if st.button('Predict'):
    try:
        # Transform input data
        scaled_input_data = scaler.transform(input_data)
        
        # Predict
        cancer_predict = model.predict(scaled_input_data)
        cancer_prob = model.predict_proba(scaled_input_data)
        
        # Print results
        st.write("Prediction Results:")
        st.write(f"Predicted Class: {cancer_predict[0]}")
        st.write(f"Predicted Probability: {cancer_prob[0][1]:.2f}")
        
        # Adjusted threshold
        if cancer_prob[0][1] >= 0.5:
            st.write("High risk of lung cancer")
        else:
            st.write("Low risk of lung cancer")
            
    except Exception as e:
        st.write(f"Error during prediction: {e}")


# Additional details or notes
st.write('Note: This prediction is based on a machine learning model and should not be used as medical advice. Always consult a healthcare provider.')

