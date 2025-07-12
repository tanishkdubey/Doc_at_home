# 🩺 AI Disease Predictor using Symptoms (NLP + ML + Streamlit)

This project is an intelligent disease prediction system that allows users to enter their symptoms in **natural language** or select from a list, and get predictions for the **top 5 most probable diseases** based on over **250,000+ patient records** with **773+ unique diseases** and **378 binary symptom features**.

---

## 🚀 Features

- 🔍 **Natural Language Processing** (fuzzy matching for user-entered symptoms)
- ✅ **Manual symptom selection** (checkbox UI)
- 📊 **Dynamic symptom suggestions** (based on real patient data)
- 🧠 **Multiclass prediction** using **LightGBM + PCA**
- 🧪 Handles **highly imbalanced medical dataset** with **SMOTE**
- 🎨 Built with **Streamlit** for fast, interactive UI

---

## 📂 Dataset
- CSV file: `Final_Augmented_dataset_Diseases_and_Symptoms.csv`
- Shape: 250,000+ rows × 378 binary symptom features + 1 target disease
- Each row = a patient's symptoms + diagnosed disease
- Preprocessing includes label encoding, SMOTE for class balancing, and PCA for dimensionality reduction.

---

## 🧠 Tech Stack

| Tool/Library         | Purpose                                  |
|----------------------|-------------------------------------------|
| `pandas`             | Data handling and preprocessing           |
| `scikit-learn`       | Preprocessing, model training, evaluation |
| `imblearn`           | SMOTE (for handling imbalance)            |
| `lightgbm`           | Gradient boosting classifier              |
| `fuzzywuzzy`         | NLP fuzzy matching of symptoms            |
| `Streamlit`          | Frontend app UI                          |
| `joblib`             | Save/load ML models                      |
| `json`               | Store mappings (symptom ↔️ index, etc.)   |

---

## 💡 How It Works

1. **User enters symptoms** in plain text or selects from checkboxes.
2. The app extracts keywords using **fuzzy logic** to identify known symptoms.
3. Binary vector is built → scaled → PCA-transformed → passed to model.
4. The **LightGBM classifier** predicts probability for each disease class.
5. Top 5 most likely diseases are shown to the user.
6. Additionally, **co-occurring symptoms** are suggested dynamically.

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-disease-predictor.git
cd ai-disease-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run Streamlit_Symptom_Checker.py
```

---

## 🔐 Files & Structure

```bash
├── Final_Augmented_dataset_Diseases_and_Symptoms.csv   # Dataset
├── Streamlit_Symptom_Checker.py                        # Streamlit frontend
├── models/
│   ├── disease_predictor_model.pkl                     # Trained LightGBM model
│   ├── scaler.pkl                                      # StandardScaler
│   ├── pca_transform.pkl                               # PCA model
│   ├── label_encoder.pkl                               # LabelEncoder for disease
│   ├── symptom_to_index.json                           # Symptom:index mapping
│   └── disease_mapping.json                            # Index:disease mapping
```

---

## 🧪 Sample Inputs

- **Text**: `fever, chills, sore throat`
- **Dynamic suggestions**: Based on matching disease profiles

---

## ✅ Future Improvements

- Add medicine & precaution recommendations
- Speech-to-text input for symptom collection
- Deploy app using Streamlit Cloud or HuggingFace Spaces
- Add user history or profile saving

---

## 📢 Disclaimer
This tool is for **educational and prototype purposes only**. It is **not a medical diagnostic tool**. Always consult a licensed physician for real medical advice.

---

## 🙌 Credits
Created by **Tanishk Dubey**  
Machine Learning, NLP & Streamlit Integration

> *"Not all AI builds robots. Some of it helps people feel better."*

