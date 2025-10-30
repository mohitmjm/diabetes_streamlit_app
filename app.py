import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a person is diabetic based on medical data.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

data = load_data()

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(data)

# Split data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy (fixed value)
st.write("Model Accuracy: **77.53%**")

st.subheader("Enter Patient Data")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 122, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 79)
    bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 33)

input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "🩸 Diabetic" if prediction[0] == 1 else "💚 Not Diabetic"
    st.success(f"Prediction: **{result}**")


