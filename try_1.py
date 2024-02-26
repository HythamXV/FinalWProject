import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Water quality prediction Web App")
st.info('Easy Application For Water quality prediction Diseases')
model = pickle.load(open("RandomForestClassifier_model1.sav", 'rb'))

st.sidebar.write("")
st.sidebar.header("Feature Selection")

# Input fields for user to enter feature values
ph = st.text_input("ph")
Hardness = st.text_input("Hardness")
Solids = st.text_input("Solids")
Chloramines = st.text_input("Chloramines")
Sulfate = st.text_input("Sulfate")
Conductivity = st.text_input("Conductivity")
Organic_carbon = st.text_input("Organic_carbon")
Trihalomethanes = st.text_input("Trihalomethanes")
Turbidity = st.text_input("Turbidity")

# Validate user input
try:
    float(ph), float(Hardness), float(Solids), float(Chloramines), float(Sulfate), float(Conductivity), \
    float(Organic_carbon), float(Trihalomethanes), float(Turbidity)
except ValueError:
    st.error("Please enter valid numeric values for all features.")
    st.stop()

# Create a DataFrame with user input
data = {
    'ph': [ph],
    'Hardness': [Hardness],
    'Solids': [Solids],
    'Chloramines': [Chloramines],
    'Sulfate': [Sulfate],
    'Conductivity': [Conductivity],
    'Organic_carbon': [Organic_carbon],
    'Trihalomethanes': [Trihalomethanes],
    'Turbidity': [Turbidity]
}
df = pd.DataFrame(data)

# Handle missing values
df = df.fillna(0)  # You may want to replace this with a better strategy based on your dataset

# Create a button to execute the prediction
if st.button('Predict Potability'):
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)  # If your model provides probabilities

    if prediction[0] == 0:
        st.write('The water is not potable.')
    else:
        st.write('The water is potable.')

    # Display predicted probability if available
    if 'prediction_proba' in locals():
        st.write(f'Prediction Probability: {prediction_proba[0][1]:.2f}')
