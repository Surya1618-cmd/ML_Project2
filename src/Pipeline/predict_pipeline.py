import os
import pickle # Used for loading preprocessor and model
import pandas as pd
import numpy as np # Needed for numerical operations
import sys # For CustomException

# Assuming CustomException is in src/exception.py
from src.exception import CustomException 
from src.logger import get_logger # For logging within the pipeline

logging = get_logger(__name__)

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.feature_columns_path = os.path.join("artifacts", "feature_columns.pkl") # Path to feature columns

        # Load trained model & preprocessor
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logging.info(f"Model loaded from {self.model_path}")

            with open(self.preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            logging.info(f"Preprocessor loaded from {self.preprocessor_path}")

            with open(self.feature_columns_path, "rb") as f:
                self.loaded_feature_columns = pickle.load(f)
            logging.info(f"Feature columns loaded from {self.feature_columns_path}: {self.loaded_feature_columns}")

        except Exception as e:
            logging.error(f"Error loading model, preprocessor, or feature columns in PredictPipeline: {e}", exc_info=True)
            raise CustomException(f"Error loading model, preprocessor, or feature columns in PredictPipeline: {e}", sys)

        # Disease-specific recommendations (from your original code)
        self.recommendations_map = { # Renamed to avoid conflict with method
            "HeartDisease": "Reduce salt intake, exercise regularly, and avoid smoking.",
            "Diabetes": "Monitor blood sugar levels, eat a balanced diet, and maintain a healthy weight.",
            "Hypertension": "Stay hydrated, reduce stress, and avoid excessive salt.",
            "Asthma": "Avoid allergens, keep an inhaler handy, and monitor air quality.",
            "KidneyDisease": "Stay hydrated and avoid excessive salt/protein.",
            "LiverDisease": "Avoid alcohol, eat liver-friendly foods, and maintain a healthy weight.",
            "Cancer": "Maintain a healthy lifestyle, avoid tobacco, and go for regular screenings.",
            "Obesity": "Follow a balanced diet and engage in regular physical activity.",
            "Arthritis": "Do light exercises, maintain joint mobility, and avoid excessive strain.",
            "COPD": "Avoid smoking, stay away from pollutants, and do breathing exercises.",
            "MentalHealthIssue": "Practice stress management, meditation, or seek counseling.",
        }
        
        self.target_columns = [ # Define target columns here for consistency
            "HeartDisease", "Diabetes", "Hypertension", "Asthma", 
            "KidneyDisease", "LiverDisease", "Cancer", "Obesity", 
            "Arthritis", "COPD", "MentalHealthIssue"
        ]

        # Store the original raw feature names here for robust input DataFrame creation
        # This list should match the columns of X_train_raw before preprocessing in data_transform.py
        self.raw_feature_names = [
            "Age", "Sex", "BMI", "Smoker", "AlcoholConsumption", "PhysicalActivity", 
            "DietQuality", "BloodPressure", "GlucoseLevel", "CholesterolLevel", "FamilyHistory"
        ]


    def predict(self, features: dict):
        # Convert input features dictionary to DataFrame
        # Ensure the DataFrame has all expected RAW input features, initialized to 0 if missing
        input_df_raw = pd.DataFrame([features])
        
        # Reindex to ensure all expected raw features are present, filling missing with 0
        df_for_preprocessor = input_df_raw.reindex(columns=self.raw_feature_names, fill_value=0)

        # --- Preprocessing steps (applied to the raw features) ---
        # 1. Fill any remaining NaNs in the raw input_df before transformation
        # This handles cases where a numerical input might be missing or coerced to NaN
        df_for_preprocessor.fillna(0, inplace=True) 

        logging.debug(f"DataFrame columns before preprocessor transform: {df_for_preprocessor.columns.tolist()}")
        logging.debug(f"DataFrame shape before preprocessor transform: {df_for_preprocessor.shape}")

        # 2. Transform features using the loaded preprocessor
        # The preprocessor expects the RAW features as input.
        transformed_features = self.preprocessor.transform(df_for_preprocessor) 
        logging.info("Features transformed by preprocessor.")
        
        # Make predictions
        preds = self.model.predict(transformed_features)        
        logging.info(f"Raw model predictions: {preds}")

        # Map predictions (continuous output from regressors acting as classifiers) to binary (0 or 1)
        # Assuming predictions are a 2D array: [[pred_for_target1, pred_for_target2, ...]]
        results = {col: (1 if preds[0][i] > 0.5 else 0) for i, col in enumerate(self.target_columns)}
        logging.info(f"Formatted prediction results: {results}")

        # Generate only relevant recommendations for high-risk diseases
        high_risk_recommendations = [
            self.recommendations_map[disease] for disease, risk in results.items() if risk == 1
        ]
        
        # If no high risks, provide general healthy living recommendations
        if not high_risk_recommendations:
            high_risk_recommendations.append("Your current health indicators suggest a low risk profile. Continue to maintain your exemplary lifestyle!")
            high_risk_recommendations.append("Schedule routine comprehensive check-ups with your healthcare provider for sustained wellness monitoring.")
        else:
            high_risk_recommendations.append("It is highly recommended to consult a healthcare professional for personalized advice based on these predictive analytics.")

        return results, high_risk_recommendations

if __name__ == "__main__":
    logging.info("Running PredictPipeline directly for testing purposes.")
    try:
        pipeline = PredictPipeline()
        
        # --- Create some dummy input features for testing ---
        # These must match the columns your Streamlit app collects
        dummy_features = {
            "Age": 45.0,
            "Sex": "Male",
            "BMI": 28.5,
            "Smoker": "No",
            "AlcoholConsumption": "Moderate",
            "PhysicalActivity": "Active",
            "DietQuality": "Good",
            "BloodPressure": "Normal",
            "GlucoseLevel": 110.0,
            "CholesterolLevel": 190.0,
            "FamilyHistory": "Yes"
        }
        
        predictions, recommendations = pipeline.predict(dummy_features)
        
        logging.info("--- TEST PREDICTION RESULTS ---")
        logging.info(f"Predictions: {predictions}")
        logging.info(f"Recommendations: {recommendations}")
        logging.info("--- END TEST ---")

    except Exception as e:
        logging.error(f"Error during direct PredictPipeline test: {e}", exc_info=True)
        print(f"An error occurred during direct PredictPipeline test: {e}")
