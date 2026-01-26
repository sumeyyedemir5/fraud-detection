from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# API Başlatma
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Bankacılık işlemleri için anlık dolandırıcılık tespiti servisi",
    version="1.0.0"
)

# Modeli Yükle
model = joblib.load('models/fraud_model.pkl')

# Veri Doğrulama Modeli (Pydantic)
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    scaled_amount: float # Preprocess aşamasında ölçeklendirdiğimiz tutar

@app.get("/")
def home():
    return {"mesaj": "Dolandırıcılık Tespit Sistemi Aktif!"}

@app.post("/tahmin")
def predict_fraud(data: Transaction):
    try:
        # Gelen veriyi DataFrame'e dönüştür (Modelin beklediği format)
        features = pd.DataFrame([data.dict()])
        # Tahmin yap
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()

        return {
            "is_fraud": int(prediction),
            "fraud_probability": round(probability[1], 4),
            "status": "DİKKAT: Dolandırıcılık Şüphesi!" if prediction == 1 else "Güvenli İşlem"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
