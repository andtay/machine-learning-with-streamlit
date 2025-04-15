import streamlit as st
from pickle import load
import pandas as pd
from datetime import datetime

@st.cache_resource
def load_model():
    with open("../models/Decision_tree_model.sav", 'rb') as f:
        return load(f)

model = load_model()


st.title('Are you diabetic?')

Pregnancies = st.slider("Pregnancies", min_value = 0.0, max_value = 17.0, step = 1.0)
Glucose = st.slider ("Glucose", min_value = 0.0, max_value = 200.0, step = 1.0)
BloodPressure = st.slider("BloodPressure", min_value = 20.0, max_value = 120.0, step = 1.0)
# Insulin = st.slider ("Insulin", min_value = 0.0,max_value= 600.0,step= 10.0)
Insulin = st.text_input('Insulin level:')
Bmi = st.slider("Bmi", min_value = 10.0, max_value = 70.0, step = 0.5)
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree level:')
Age = st.text_input('Age:')

# Mostrar los datos antes de predecir
st.subheader("🧾 Your Input Data:")
input_dict = {
    "Pregnancies": Pregnancies,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "Insulin": Insulin,
    "BMI": Bmi,
    "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    "Age": Age
}
input_df = pd.DataFrame([input_dict])
st.dataframe(input_df)

if st.button("Predict"):
    data_a_predecir = [[Pregnancies, Glucose, BloodPressure, Insulin, Bmi, 
                        DiabetesPedigreeFunction, Age]]
    prediction = model.predict(data_a_predecir)[0]
    if prediction == 0:
        prediction = 'NO'
    else:
        prediction = 'YES'

    st.write("Prediction", prediction)
     # Mostrar explicación básica
    st.info("🤖 This prediction was made using a Decision Tree model trained on medical data.")
    
    # Guardar los resultados para descarga
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_df["Prediction"] = ["Diabetic" if prediction == 1 else "Not Diabetic"]
    
    csv = input_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv,
        file_name=f"diabetes_prediction_{timestamp}.csv",
        mime='text/csv'
    )