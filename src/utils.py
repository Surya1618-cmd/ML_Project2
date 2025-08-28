import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier # Ensure this is imported

from src.exception import CustomException
from src.logger import get_logger

logging = get_logger(__name__)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}", exc_info=True)
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        logging.info("Starting model evaluation in utils.py...")
        report = {}

        for model_name, model_base in models.items():
            logging.info(f"Evaluating model: {model_name}")
            
            # Always wrap the base model in MultiOutputClassifier for evaluation
            # Then apply GridSearchCV to this MultiOutputClassifier instance
            multi_output_model = MultiOutputClassifier(model_base)

            # Check if hyperparameters exist for the current model
            if model_name in params and params[model_name]:
                hyperparams = params[model_name]
                logging.info(f"Applying GridSearchCV for {model_name} with params: {hyperparams}")
                
                # GridSearchCV now tunes the MultiOutputClassifier with estimator__ prefixed params
                gs = GridSearchCV(estimator=multi_output_model, param_grid=hyperparams, cv=3, n_jobs=-1, verbose=0, error_score='raise')
                gs.fit(X_train, y_train)
                
                # gs.best_estimator_ will be the best MultiOutputClassifier
                final_model_for_eval = gs.best_estimator_
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            else:
                # If no params, fit the MultiOutputClassifier directly
                final_model_for_eval = multi_output_model
                final_model_for_eval.fit(X_train, y_train)
                logging.info(f"No specific hyperparameters for {model_name}, using default MultiOutputClassifier.")

            # Make predictions
            y_train_pred = final_model_for_eval.predict(X_train)
            y_test_pred = final_model_for_eval.predict(X_test)

            # Calculate F1-score for multi-output classification
            train_model_score = f1_score(y_train, y_train_pred, average='macro')
            test_model_score = f1_score(y_test, y_test_pred, average='macro')

            report[model_name] = test_model_score
            logging.info(f"Model {model_name} - Train F1-score: {train_model_score:.4f}, Test F1-score: {test_model_score:.4f}")

        logging.info("Model evaluation completed.")
        return report

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}", exc_info=True)
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object successfully loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}", exc_info=True)
        raise CustomException(e, sys)