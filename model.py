import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json
from pathlib import Path


class DiseasePredictor:
    def __init__(self, config):
        self.config = config
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.config['data_path'])

        # Filter rare diseases
        disease_counts = df["diseases"].value_counts()
        common_diseases = disease_counts[disease_counts >= self.config['min_samples']].index
        self.df_filtered = df[df["diseases"].isin(common_diseases)]

        # Store disease mapping
        self.disease_mapping = {i: disease for i, disease in enumerate(common_diseases)}
        with open(self.models_dir / "disease_mapping.json", "w") as f:
            json.dump(self.disease_mapping, f, indent=4)

        return self.df_filtered.drop(columns=["diseases"]), self.df_filtered["diseases"]

    def preprocess_data(self, X, y):
        """Split, encode, and balance the dataset"""
        print("Preprocessing data...")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            stratify=y,
            random_state=self.config['random_state']
        )

        # Label encoding
        self.le = LabelEncoder()
        y_train_encoded = self.le.fit_transform(y_train)
        y_test_encoded = self.le.transform(y_test)

        # Handle class imbalance with SMOTE
        if self.config['use_smote']:
            smote = SMOTE(
                sampling_strategy="minority",
                random_state=self.config['random_state'],
                k_neighbors=min(3, self.config['min_samples'] - 1)  # Adjust for small classes
            )
            X_train, y_train_encoded = smote.fit_resample(X_train, y_train_encoded)

        return X_train, X_test, y_train_encoded, y_test_encoded

    def feature_engineering(self, X_train, X_test):
        """Scale and reduce dimensionality of features"""
        print("Engineering features...")

        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Dimensionality reduction
        if self.config['use_pca']:
            self.pca = PCA(n_components=self.config['pca_components'])
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)

        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, y_train, X_test, y_test):
        """Train and evaluate the LightGBM model"""
        print("Training model...")

        self.model = LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=self.config['num_leaves'],
            max_depth=self.config['max_depth'],
            learning_rate=self.config['learning_rate'],
            n_estimators=self.config['n_estimators'],
            min_child_samples=self.config['min_child_samples'],
            objective='multiclass',
            num_class=len(self.le.classes_),
            metric='multi_logloss',
            class_weight='balanced',
            random_state=self.config['random_state']
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config['early_stopping_rounds']),
                lgb.log_evaluation(period=50)
            ]
        )

        return self.model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        y_pred = model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0, target_names=self.le.classes_))

        print(f"\nF1 Score (Macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    def save_components(self):
        """Save all model components"""
        print("Saving model components...")
        joblib.dump(self.model, self.models_dir / "disease_predictor_model.pkl")
        joblib.dump(self.le, self.models_dir / "label_encoder.pkl")

        if self.config['use_pca']:
            joblib.dump(self.pca, self.models_dir / "pca_transform.pkl")

        joblib.dump(self.scaler, self.models_dir / "scaler.pkl")
        print("✅ All components saved successfully!")


if __name__ == "__main__":
    # Configuration dictionary
    config = {
        'data_path': "Final_Augmented_dataset_Diseases_and_Symptoms.csv",
        'min_samples': 10,
        'test_size': 0.3,
        'random_state': 42,
        'use_smote': True,
        'use_pca': True,
        'pca_components': 100,
        'num_leaves': 31,
        'max_depth': 10,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 20,
        'early_stopping_rounds': 10
    }

    # Initialize and run pipeline
    pipeline = DiseasePredictor(config)

    # Load and preprocess data
    X, y = pipeline.load_data()

    # Split and balance data
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(X, y)

    # Feature engineering
    X_train_scaled, X_test_scaled = pipeline.feature_engineering(X_train, X_test)

    # Train model
    model = pipeline.train_model(X_train_scaled, y_train, X_test_scaled, y_test)

    # Evaluate model
    pipeline.evaluate_model(model, X_test_scaled, y_test)

    # Save components
    pipeline.save_components()