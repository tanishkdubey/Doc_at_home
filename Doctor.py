# Streamlit Disease Predictor App with NLP, Fuzzy Matching, and Dynamic Checkbox Suggestions

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from fuzzywuzzy import process, fuzz
from collections import Counter

# Load Models and Components
model = joblib.load("models/disease_predictor_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")
pca = joblib.load("models/pca_transform.pkl")

# Load symptom mapping
with open("models/symptom_to_index.json") as f:
    symptom_to_index = json.load(f)

index_to_symptom = {v: k for k, v in symptom_to_index.items()}
all_symptoms = list(symptom_to_index.keys())

# Load dataset for dynamic suggestions
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

# Streamlit UI Setup
st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🩺 AI-Powered Disease Predictor")
st.markdown("Enter your symptoms in text OR select them manually. Then click Predict to get possible diseases.")

# --- Input Handling ---
user_input = st.text_area("Type your symptoms (comma separated or natural language):", "e.g., headache, body pain, fever")
manual_symptoms = st.multiselect("Or select symptoms manually:", options=all_symptoms)

# --- NLP + Fuzzy Matching ---
def extract_symptoms(text, threshold=80):
    tokens = [t.strip().lower() for t in text.replace(",", ";").replace("and", ";").split(";")]
    found = []
    for token in tokens:
        match, score = process.extractOne(token, all_symptoms, scorer=fuzz.token_set_ratio)
        if score >= threshold:
            found.append(match)
    return list(set(found))

nlp_symptoms = extract_symptoms(user_input) if user_input else []
found_symptoms = list(set(nlp_symptoms + manual_symptoms))

# --- Dynamic Symptom Suggestion Logic ---
def get_dynamic_symptom_suggestions(found_symptoms):
    df_filtered = df.copy()
    for symptom in found_symptoms:
        if symptom in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[symptom] == 1]
    counter = Counter()
    for col in df.drop(columns=["diseases"]).columns:
        if col not in found_symptoms:
            count = df_filtered[col].sum()
            if count > 0:
                counter[col] = count
    return [s for s, _ in counter.most_common(10)]

suggestions = get_dynamic_symptom_suggestions(found_symptoms)

if suggestions:
    st.write("\n💡 **People with your symptoms also reported:**")
    extra_symptoms = st.multiselect("Add more if applicable:", options=suggestions)
    all_selected_symptoms = list(set(found_symptoms + extra_symptoms))
else:
    all_selected_symptoms = found_symptoms

# --- Prediction ---
if st.button("🔍 Predict Disease"):
    if not all_selected_symptoms:
        st.warning("Please enter or select at least one symptom.")
    else:
        # Create binary input vector
        input_vector = np.zeros((1, len(symptom_to_index)))
        for symptom in all_selected_symptoms:
            if symptom in symptom_to_index:
                idx = symptom_to_index[symptom]
                input_vector[0, idx] = 1

        # Scale & PCA
        scaled = scaler.transform(input_vector)
        reduced = pca.transform(scaled)

        # Predict
        probs = model.predict_proba(reduced)[0]
        top_indices = np.argsort(probs)[::-1][:5]
        top_diseases = [(le.inverse_transform([i])[0], probs[i]) for i in top_indices]

        st.subheader("🧾 Top Predicted Diseases:")
        for disease, prob in top_diseases:
            st.markdown(f"- **{disease}** — Confidence: `{prob*100:.2f}%`")

        st.info("⚠️ This prediction is AI-based and not a medical diagnosis. Always consult a doctor.")
