import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import dill # For loading the model
# Removed LabelEncoder as it's now handled within PredictPipeline

# Assuming CustomException and get_logger are in src/
from src.exception import CustomException
from src.logger import get_logger
# New: Import PredictPipeline
from src.Pipeline.predict_pipeline import PredictPipeline 

logging = get_logger(__name__)

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="PERSONALIZED HEALTH ASSISTANT", layout="wide", initial_sidebar_state="collapsed")

# --- Global Variables for Model and Preprocessor ---
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl") # New: Preprocessor path

# Target columns (must match what was used during training from health_dataset_with_issues.csv)
TARGET_COLUMNS = ["HeartDisease", "Diabetes", "Hypertension", "Asthma", 
                  "KidneyDisease", "LiverDisease", "Cancer", "Obesity", 
                  "Arthritis", "COPD", "MentalHealthIssue"]

# --- Recommendation Logic ---
def generate_recommendations(predictions_dict):
    """Generates health recommendations based on predicted risk levels."""
    recs = []
    has_high_risk = False
    
    # Define a threshold for "high risk" from the regression output
    # For R2, a prediction closer to 1 (Yes) means higher risk.
    # Let's say a prediction > 0.5 is considered "High Risk" for display.
    RISK_THRESHOLD = 0.5 

    for disease, score in predictions_dict.items():
        if score > RISK_THRESHOLD:
            has_high_risk = True
            if disease == "HeartDisease":
                recs.append("Consult a cardiologist for heart health assessment. Focus on a heart-healthy diet and regular exercise.")
            elif disease == "Diabetes":
                recs.append("Monitor blood sugar levels diligently. Adopt a low-glycemic diet and increase physical activity.")
            elif disease == "Hypertension":
                recs.append("Regularly check blood pressure. Reduce sodium intake and manage stress effectively.")
            elif disease == "Asthma":
                recs.append("Identify and avoid environmental triggers. Keep emergency medication readily accessible.")
            elif disease == "KidneyDisease":
                recs.append("Manage blood pressure and diabetes. Reduce sodium and processed food intake.")
            elif disease == "LiverDisease":
                recs.append("Significantly reduce or abstain from alcohol. Maintain a healthy weight and balanced diet.")
            elif disease == "Cancer":
                recs.append("Discuss personalized screening options and preventative lifestyle changes with your oncologist.")
            elif disease == "Obesity":
                recs.append("Prioritize sustainable weight management through tailored diet and exercise plans. Consider professional guidance.")
            elif disease == "Arthritis":
                recs.append("Engage in low-impact exercises to maintain joint flexibility. Explore anti-inflammatory diets.")
            elif disease == "COPD":
                recs.append("Cease smoking immediately and avoid all forms of secondhand smoke. Consult a pulmonologist for comprehensive respiratory management.")
            elif disease == "MentalHealthIssue":
                recs.append("Seek consultation with a mental health professional. Integrate stress-reduction techniques like mindfulness and meditation.")
            
    if not has_high_risk:
        recs.append("Your current health indicators suggest a low risk profile. Continue to maintain your exemplary lifestyle!")
        recs.append("Schedule routine comprehensive check-ups with your healthcare provider for sustained wellness monitoring.")
    else:
        recs.append("Immediate consultation with a healthcare professional is highly recommended for a personalized diagnostic and intervention plan based on these predictive analytics.")
    return recs

# --- Model Loading (using Streamlit's caching) ---
@st.cache_resource
def load_predict_pipeline():
    """Loads the PredictPipeline instance."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            pipeline = PredictPipeline()
            logging.info("PredictPipeline loaded successfully.")
            return pipeline
        else:
            logging.error(f"Model ({MODEL_PATH}) or Preprocessor ({PREPROCESSOR_PATH}) file not found.")
            st.warning("System Alert: Core Predictive Module or Preprocessor Not Detected. Please Initiate Model Training Protocol.")
            return None
    except Exception as e:
        logging.error(f"Error loading PredictPipeline: {e}", exc_info=True)
        st.error(f"System Alert: Error Loading Predictive Module. Details: {e}")
        return None

# Load the PredictPipeline instance once
predict_pipeline_instance = load_predict_pipeline()

# Custom CSS for Iron Man theme
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    .main {{
        background-color: #0d1117; /* Dark background */
        color: #e6edf3; /* Light text */
        font-family: 'Share Tech Mono', monospace; /* Futuristic font */
    }}
    h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm {{ /* Target Streamlit headers and titles */
        font-family: 'Orbitron', sans-serif; /* Iron Man style font */
        color: #00ffff; /* Cyan glow */
        text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff, 0 0 15px #00ffff;
    }}
    .stButton>button {{
        background-image: linear-gradient(to right, #00ffff, #007bff); /* Cyan to blue gradient */
        color: #0d1117; /* Dark text on button */
        font-weight: bold;
        border-radius: 0.5rem;
        border: 1px solid #00ffff; /* Cyan border */
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px #00ffff; /* Cyan glow */
        font-family: 'Orbitron', sans-serif;
    }}
    .stButton>button:hover {{
        transform: scale(1.05);
        background-image: linear-gradient(to right, #00e5e5, #0056b3);
        box-shadow: 0 0 15px #00ffff, 0 0 20px #00ffff;
    }}
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input {{
        background-color: #161b22; /* Darker input background */
        color: #00ffff; /* Cyan text */
        border-radius: 0.5rem;
        border: 1px solid #00ffff; /* Cyan border */
        box-shadow: 0 0 5px #00ffff; /* Cyan glow */
        padding: 0.5rem 0.75rem;
        font-family: 'Share Tech Mono', monospace;
    }}
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stNumberInput>div>div>input:focus {{
        border-color: #00e5e5;
        box-shadow: 0 0 8px #00e5e5, 0 0 12px #00e5e5;
    }}
    .stDataFrame {{
        border: 1px solid #00ffff; /* Cyan border for table */
        border-radius: 0.5rem;
        box-shadow: 0 0 10px #00ffff;
    }}
    .stDataFrame table thead th {{
        background-color: #00ffff !important; /* Cyan header */
        color: #0d1117 !important; /* Dark text on cyan */
        font-family: 'Orbitron', sans-serif;
    }}
    .stDataFrame table tbody tr {{
        background-color: #161b22; /* Darker row background */
        color: #e6edf3; /* Light text */
    }}
    .stDataFrame table tbody tr:nth-child(even) {{
        background-color: #0d1117; /* Alternating dark row */
    }}
    .stDataFrame table tbody td {{
        border-color: #00ffff; /* Cyan border for cells */
    }}
    .risk-high {{
        background-color: #ff0000; /* Red */
        color: white;
        font-weight: bold;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        box-shadow: 0 0 8px #ff0000;
    }}
    .risk-low {{
        background-color: #00ff00; /* Green */
        color: #0d1117;
        font-weight: bold;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        box-shadow: 0 0 8px #00ff00;
    }}
    .stAlert {{
        background-color: #161b22;
        border: 1px solid #00ffff;
        box-shadow: 0 0 10px #00ffff;
        color: #00ffff;
    }}
    .stAlert > div > div > div > p {{
        font-family: 'Share Tech Mono', monospace;
    }}
    </style>
    """, unsafe_allow_html=True)

st.title("Healthcare Predictive Diagnostic System")
st.markdown("### Predictive Diagnostic System - Mark I")

if predict_pipeline_instance is None:
    st.error("System Offline: Core Predictive Module or Feature Definitions Not Detected. Please Initiate Model Training Protocol.")
else:
    with st.form("health_input_form", clear_on_submit=False):
        st.markdown("#### Input Biometric & Lifestyle Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (Years):", min_value=1, max_value=120, value=30, step=1)
            bmi = st.text_input("BMI (kg/m²):", value="25.0")
            smoker = st.selectbox("Smoking Status:", ["Yes", "No", "Former Smoker"])
            diet_quality = st.selectbox("Dietary Regimen Quality:", ["Good", "Average", "Poor"])
        with col2:
            sex = st.selectbox("Biological Sex:", ["Male", "Female", "Other"])
            glucose_level = st.text_input("Glucose Level (mg/dL):", value="100.0")
            alcohol_consumption = st.selectbox("Alcohol Consumption Frequency:", ["High", "Moderate", "Low", "None"])
            blood_pressure = st.selectbox("Blood Pressure Status:", ["Normal", "Prehypertension", "Hypertension"])
        with col3:
            cholesterol_level = st.text_input("Cholesterol Level (mg/dL):", value="180.0")
            physical_activity = st.selectbox("Physical Activity Level:", ["Active", "Moderate", "Sedentary"])
            family_history = st.selectbox("Genetic Predisposition (Family History):", ["Yes", "No"])
        
        st.markdown("---")
        submitted = st.form_submit_button("ANALYZE BIOMETRICS")

        if submitted:
            with st.spinner("Processing Data... Running Predictive Algorithms..."):
                try:
                    input_features = { # Renamed to input_features to match PredictPipeline
                        "Age": float(age),
                        "Sex": sex,
                        "BMI": float(bmi),
                        "Smoker": smoker,
                        "AlcoholConsumption": alcohol_consumption,
                        "PhysicalActivity": physical_activity,
                        "DietQuality": diet_quality,
                        "BloodPressure": blood_pressure,
                        "GlucoseLevel": float(glucose_level),
                        "CholesterolLevel": float(cholesterol_level),
                        "FamilyHistory": family_history
                    }
                    
                    # Call the predict method of the PredictPipeline instance
                    predictions_dict, recommendations = predict_pipeline_instance.predict(input_features)
                    
                    st.markdown("---")
                    st.subheader("Predictive Analytics Report")
                    
                    results_df = pd.DataFrame(predictions_dict.items(), columns=["Health Metric", "Risk Assessment"])
                    results_df["Risk Assessment"] = results_df["Risk Assessment"].map({1: "High Risk", 0: "Low Risk"})
                    
                    # Custom display for DataFrame
                    st.dataframe(results_df, hide_index=True)
                    
                    st.markdown("---")
                    st.subheader("Operational Directives (Recommendations)")
                    for rec in recommendations:
                        st.markdown(f"✅ <span style='font-family: \"Share Tech Mono\", monospace; color: #00ffff;'>{rec}</span>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Critical System Failure: Prediction Protocol Aborted. Error: {e}")
                    logging.error(f"Error during Streamlit prediction: {e}", exc_info=True)