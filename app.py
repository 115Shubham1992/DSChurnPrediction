from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Telco Churn Prediction API")

try:
    model = joblib.load('model/churn_model.pkl')
except Exception as e:
    model = None

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        data = pd.DataFrame([customer.dict()])

        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        data['Total_Services'] = data[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
        data['Is_Auto_Payment'] = data['PaymentMethod'].apply(lambda x: 1 if 'automatic' in x.lower() else 0)

        prediction_val = model.predict(data)[0]
        prediction_prob = model.predict_proba(data)[0][1]

        result = "Yes" if prediction_val == 1 else "No"
        
        return {
            "prediction": result,
            "churn_probability": round(float(prediction_prob), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {
        "service": "Telco Churn Prediction API",
        "model_loaded": model is not None,
    }

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}