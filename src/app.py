# import package
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib
import datetime

# 1. Initialize the API
app = FastAPI(
    title="Banking AI Decision Engine",
    description="Live API for predicting credit default risk using XGBoost and hardcoded banking rules.",
    version="2.0"
)

# 2. Load the Production Pipeline
try:
    # Adjust this path based on where you run the server
    model_pipeline = joblib.load('models/xgboost_model_pipeline.pkl') 
except Exception as e:
    model_pipeline = None
    print(f"Warning: Model not found. Please ensure the .pkl file is in the correct directory. {e}")

# 3. Define the Expected JSON Payload
class LoanApplication(BaseModel):
    customer_id: str
    features: Dict[str, float]

# 4. Define the API Endpoint
@app.post("/predict")
def predict_credit_risk(application: LoanApplication):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Machine Learning model is not loaded on the server.")
        
    # Convert JSON features into a Pandas DataFrame (1 row)
    app_data = pd.DataFrame([application.features])
    
    # --- Execute Business Rules ---
    if app_data['avg_payment_delay'].values[0] >= 2.0:
        return {
            "customer_id": application.customer_id,
            "decision": "REJECT",
            "reason": "Policy Rule: Severe Delinquency",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    if app_data['utilization_ratio'].values[0] >= 0.95:
        return {
            "customer_id": application.customer_id,
            "decision": "REJECT",
            "reason": "Policy Rule: Credit Exhaustion",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    # --- Execute ML Pipeline ---
    ml_prob = model_pipeline.predict_proba(app_data)[:, 1][0]
    
    if ml_prob >= 0.42:
        return {
            "customer_id": application.customer_id,
            "decision": "REJECT",
            "reason": "ML Model: High Default Risk",
            "probability": round(float(ml_prob), 4),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    return {
        "customer_id": application.customer_id,
        "decision": "APPROVE",
        "reason": "Passed Policy and ML Risk Check",
        "probability": round(float(ml_prob), 4),
        "timestamp": datetime.datetime.now().isoformat()
    }
