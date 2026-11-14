import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Logistic_regression_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease prediction")
st.markdown("This app predicts the likelihood of heart disease based on user input.")

age= st.slider("Age",18,100,40)
sex = st.selectbox("Sex",['M',"F"])
chest_pain = st.selectbox("Chest Pain type",["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("resting blood pressure (mm Hg) : ",80,200,120)
cholesterol = st.number_input("serum cholestrol in mg/dl : ",100,600,200)
fastingBS = st.selectbox("Fasting Blood Suger > 120 mg/dl",[0,1])
resing_ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
excercise_angina = st.selectbox("Excercise-Induced Angina",["Y","N"])
oldpeak = st.slider("Oldpeak(ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope",["UP","Flat","Down"])


if st.button("Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fastingBS,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resing_ecg: 1,
        'ExerciseAngina_' + excercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High risk of Heart Disease")
    else:
        st.success("Low risk of Heart Disease")

