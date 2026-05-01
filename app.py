import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean dataset
if "Unnamed: 133" in df.columns:
    df = df.drop("Unnamed: 133", axis=1)

# Target column
target = "Disease"

# Get symptom list
symptoms_list = [col for col in df.columns if col != target]

# Page settings
st.set_page_config(page_title="AI Disease Predictor", layout="centered")

# Title
st.title("🩺 AI Disease Prediction System")
st.write("Select symptoms and predict possible disease")

# Sidebar
st.sidebar.title("Project Info")
st.sidebar.info("This system predicts diseases using Machine Learning (Random Forest Algorithm).")

# Symptom selection
selected_symptoms = st.multiselect("Select Symptoms:", symptoms_list)

# Convert input into model format
input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]

# Prediction
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom")
    else:
        prediction = model.predict([input_data])
        disease = prediction[0]

        st.success(f"🧾 Predicted Disease: {disease}")

        # Top 3 predictions
        try:
            probabilities = model.predict_proba([input_data])[0]
            top3 = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]

            st.write("### 🔝 Top 3 Possible Diseases:")
            for dis, prob in top3:
                st.write(f"{dis} ({round(prob*100,2)}%)")
        except:
            st.info("Top predictions not available")

        # Warning
        st.warning("⚠️ This is for educational purposes only. Please consult a doctor.")