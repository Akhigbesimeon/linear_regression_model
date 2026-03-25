"""
FastAPI application for Student GPA prediction.
"""
import os
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student GPA Predictor API",
    description="API to predict student GPA based on 4 core habits.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../linear_regression/best_gpa_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../linear_regression/gpa_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "../linear_regression/gpa_feature_columns.pkl")

MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
SCALER = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
FEATURE_COLUMNS = list(joblib.load(FEATURES_PATH)) if os.path.exists(FEATURES_PATH) else []


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    """Automatically redirects the base URL to the Swagger UI."""
    return RedirectResponse(url="/docs")


class PredictionInput(BaseModel):
    """6 core student habit features used for GPA prediction."""
    study_hours: float = Field(..., ge=0.0, le=12.0, example=5.0, description="Hours studied per day (0-12)")
    screen_time: float = Field(..., ge=0.0, le=20.0, example=6.0, description="Total screen time in hours (0-20)")
    concentration: float = Field(..., ge=1.0, le=10.0, example=7.0, description="Concentration score (1-10)")
    procrastination_score: float = Field(..., ge=1.0, le=10.0, example=4.0, description="Procrastination score (1-10, lower is better)")
    backlogs: int = Field(..., ge=0, le=10, example=1, description="Number of failed/pending subjects (0-10)")
    part_time_hours: float = Field(..., ge=0.0, le=11.0, example=3.0, description="Hours worked part-time per week (0-11)")


class RetrainInput(PredictionInput):
    """Schema for training data, adding the target GPA."""
    gpa: float = Field(..., ge=0.0, le=4.0, description="Actual GPA on 0-4 scale")


class RetrainRequest(BaseModel):
    """Schema for a batch of retraining data."""
    data: List[RetrainInput] = Field(..., min_length=10)


@app.post("/predict", summary="Predict Student GPA")
def predict_gpa(input_data: PredictionInput):
    """Predicts a student's GPA based on their habits."""
    if MODEL is None or SCALER is None or len(FEATURE_COLUMNS) == 0:
        raise HTTPException(
            status_code=503, detail="Server artifacts not fully loaded. Check file paths."
        )

    try:
        input_df = pd.DataFrame([input_data.model_dump()])

        for col in FEATURE_COLUMNS:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[FEATURE_COLUMNS]

        x_scaled = SCALER.transform(input_df)
        raw_prediction = float(MODEL.predict(x_scaled)[0])

        predicted_gpa = round(max(0.0, min(4.0, raw_prediction)), 2)
        return {"predicted_gpa": predicted_gpa}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}") from e


@app.post("/retrain", summary="Retrain the Model")
def retrain_model(request: RetrainRequest):
    """Retrains the model with a new batch of data."""
    if MODEL is None or SCALER is None or len(FEATURE_COLUMNS) == 0:
        raise HTTPException(status_code=503, detail="Artifacts missing. Cannot retrain.")

    try:
        raw_data = [item.model_dump() for item in request.data]
        df = pd.DataFrame(raw_data)

        for col in FEATURE_COLUMNS:
            if col not in df.columns and col != 'gpa':
                df[col] = 0

        x_new = df[FEATURE_COLUMNS]
        y_new = df['gpa']
        x_new_scaled = SCALER.transform(x_new)
        MODEL.fit(x_new_scaled, y_new)
        joblib.dump(MODEL, MODEL_PATH)

        return {
            "status": "success",
            "message": f"Retrained on {len(df)} records."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Retraining error: {str(e)}") from e
