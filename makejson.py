import pandas as pd
import json

# Load your dataset
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

# Get all feature names (excluding the target column)
features = list(df.drop(columns=["diseases"]).columns)

# Create dictionary: {"symptom_name": index}
feature_index_dict = {feature: i for i, feature in enumerate(features)}

# Save to JSON
with open("models/symptom_to_index.json", "w") as f:
    json.dump(feature_index_dict, f, indent=4)

print("✅ Saved symptom_to_index.json")
