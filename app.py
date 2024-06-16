import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prediction import get_prediction, ScaleData

# Load the model
model = joblib.load('./model.pkl')
scaler = joblib.load('./scaler.pkl')

# Set Streamlit page configuration
st.set_page_config(page_title='Survival Prediction App', page_icon=':hospital:', layout='wide', initial_sidebar_state='expanded')

st.markdown("<h1 style='text-align: center; color: white;'>Patient Survival Prediction App</h1>", unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):
        st.subheader('Enter Patient Details:')

        d1_sysbp_noninvasive_min = st.slider(
            'The patients lowest systolic blood pressure during the first 24 hours of their unit stay, non-invasively measured', 
            41, 160, format='%d mmHg', key='d1_sysbp_noninvasive_min'
        )
        d1_sysbp_min = st.slider(
            'The patients lowest systolic blood pressure during the first 24 hours of their unit stay, invasively measured', 
            41, 160,  format='%d mmHg', key='d1_sysbp_min'
        )
        d1_spo2_min = st.slider(
            'The patients lowest peripheral capillary oxygen saturation during the first 24 hours of their unit stay', 
            0, 100,  format='%d', key='d1_spo2_min'
        )
        d1_temp_min = st.slider(
            'The patients lowest temperature during the first 24 hours of their unit stay', 
            30, 45,  format='%d', key='d1_temp_min'
        )
        ventilated_apache = st.selectbox(
            'Whether the patient was mechanically ventilated at the time of the highest scoring arterial blood gas', 
            ('Yes', 'No'), key='ventilated_apache'
        )
        gcs_verbal_apache = st.slider(
            'The Glasgow Coma Scale score component representing the best verbal response', 
            1, 5, format='%d', key='gcs_verbal_apache'
        )
        gcs_eyes_apache = st.slider(
            'The Glasgow Coma Scale score component representing the best motor response', 
            1, 4, format='%d', key='gcs_eyes_apache'
        )
        gcs_motor_apache = st.slider(
            'The Glasgow Coma Scale score component representing the best eye response', 
            1, 6,  format='%d', key='gcs_motor_apache'
        )
        apache_4a_icu_death_prob = st.slider(
            'The probability of death as predicted by the APACHE IVa model', 
            0.0, 1.0, format='%f', key='apache_4a_icu_death_prob'
        )
        apache_4a_hospital_death_prob = st.slider(
            'The probability of death as predicted by the APACHE IVa model', 
            0.0, 1.0,  format='%f', key='apache_4a_hospital_death_prob'
        )

        submit = st.form_submit_button('Predict')

    if submit:
        # Convert categorical to numerical
        ventilated_apache = 1 if ventilated_apache == 'Yes' else 0

        # Prepare data for prediction
        data_done = np.array([
            d1_sysbp_noninvasive_min, d1_sysbp_min, d1_spo2_min, d1_temp_min,
            ventilated_apache, gcs_verbal_apache, gcs_eyes_apache, 
            gcs_motor_apache, apache_4a_icu_death_prob, apache_4a_hospital_death_prob
        ]).reshape(1, -1)

        st.write('Data for prediction:', data_done)
        
        # Scale data
        data = scaler.transform(data_done)
        st.write('Scaled data:', data)
        
        # Get prediction
        prediction = get_prediction(model, data)
        st.write('Raw prediction:', prediction)
        
        # Interpret prediction
        if prediction[0] > 0.5:
            st.write(f'The patient is at risk of death ({(prediction[0] * 100).round(2)}%)')
        else:
            st.write(f'The patient is not at risk of death ({(prediction[0] * 100).round(2)}%)')

if __name__ == '__main__':
    main()
