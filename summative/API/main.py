"""
FastAPI application for Student GPA prediction and model retraining.
Optimized for low-memory environments.
"""
import os
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student GPA Predictor API",
    description="API to predict student GPA and retrain the regression model.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"

# Load models if they exist
MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
SCALER = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None


class PredictionInput(BaseModel):
    """Schema for individual student data used in prediction."""
    study_hours: float = Field(..., ge=0.0, le=24.0, description="Hours studied per week")
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Hours slept per night")
    attendance_rate: float = Field(
        ..., ge=0.0, le=100.0, description="Class attendance percentage"
    )
    extracurriculars: int = Field(
        ..., ge=0, le=20, description="Number of extracurricular activities"
    )


class RetrainInput(PredictionInput):
    """Schema for training data, including the target GPA."""
    gpa: float = Field(..., ge=0.0, le=4.0, description="Actual Student GPA")


class RetrainRequest(BaseModel):
    """Schema for a batch of retraining data."""
    data: List[RetrainInput] = Field(
        ..., min_length=10, description="Batch of new data for retraining"
    )


@app.post("/predict", summary="Predict Student GPA")
def predict_gpa(input_data: PredictionInput):
    """Takes student habits and returns a predicted GPA."""
    if MODEL is None or SCALER is None:
        raise HTTPException(
            status_code=503, detail="Model or scaler not loaded on the server."
        )

    try:
        # Extract data into a 2D numpy array instead of a pandas DataFrame
        # IMPORTANT: The order must match how the model was originally trained
        features = np.array([[
            input_data.study_hours,
            input_data.sleep_hours,
            input_data.attendance_rate,
            input_data.extracurriculars
        ]])

        x_scaled = SCALER.transform(features)
        prediction = MODEL.predict(x_scaled)[0]
        final_gpa = max(0.0, min(4.0, float(prediction)))

        return {"predicted_gpa": round(final_gpa, 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}") from e


@app.post("/retrain", summary="Retrain the Model")
def retrain_model(request: RetrainRequest):
    """Retrains the model using a new batch of uploaded data."""
    if MODEL is None or SCALER is None:
        raise HTTPException(
            status_code=503, detail="Base model not found. Cannot retrain."
        )

    try:
        raw_data = [item.model_dump() for item in request.data]
        
        # Build 2D numpy array for features and 1D array for target
        x_new = np.array([
            [d['study_hours'], d['sleep_hours'], d['attendance_rate'], d['extracurriculars']]
            for d in raw_data
        ])
        y_new = np.array([d['gpa'] for d in raw_data])

        x_new_scaled = SCALER.transform(x_new)
        MODEL.fit(x_new_scaled, y_new)

        joblib.dump(MODEL, MODEL_PATH)

        return {
            "status": "success",
            "message": f"Model successfully retrained on {len(raw_data)} new records."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Retraining error: {str(e)}") from e