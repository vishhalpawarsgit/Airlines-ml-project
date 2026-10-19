"""FastAPI service that exposes the saved arrival-delay model."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "arrival_delay_model.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
EXPECTED_FEATURES = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

model: Pipeline | None = None
model_metadata: dict[str, object] = {}


class PredictionRequest(BaseModel):
    """Validated input fields used by the trained after-departure model."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=0, le=120)
    flight_distance: float = Field(ge=0, le=20_000)
    departure_delay_minutes: float = Field(ge=0, le=10_000)
    gender: Literal["Female", "Male"]
    customer_type: Literal["Loyal Customer", "disloyal Customer"]
    type_of_travel: Literal["Personal Travel", "Business travel"]
    flight_class: Literal["Business", "Eco", "Eco Plus"]


class PredictionResponse(BaseModel):
    predicted_arrival_delay_minutes: float
    prediction_stage: Literal["after-departure"]
    advisory: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the trusted local model once when the API starts."""
    global model, model_metadata

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Trained model not found at {MODEL_PATH}. Run `python train.py` first."
        )
    if not METRICS_PATH.is_file():
        raise RuntimeError(
            f"Model metadata not found at {METRICS_PATH}. Run `python train.py` first."
        )

    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        model_metadata = json.load(metrics_file)
    if (
        model_metadata.get("prediction_stage") != "after-departure"
        or model_metadata.get("features") != EXPECTED_FEATURES
    ):
        raise RuntimeError(
            "The saved model is not the expected after-departure model. "
            "Run `python train.py` with its default prediction stage."
        )

    # Only load model artifacts created locally by this project's training script.
    model = joblib.load(MODEL_PATH)
    yield
    model = None
    model_metadata = {}


app = FastAPI(
    title="Airline Arrival Delay API",
    description="Predict arrival delay after departure using the trained project model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, object]:
    """Report service readiness and the saved test-set MAE."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    tuned_metrics = model_metadata.get("tuned_hist_gradient_boosting", {})
    mae = (
        tuned_metrics.get("mae_minutes")
        if isinstance(tuned_metrics, dict)
        else None
    )
    return {
        "status": "ok",
        "prediction_stage": "after-departure",
        "test_mae_minutes": mae,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Validate a flight record, invoke the model, and return delay minutes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    model_input = pd.DataFrame(
        [
            {
                "Age": request.age,
                "Flight Distance": request.flight_distance,
                "Departure Delay in Minutes": request.departure_delay_minutes,
                "Gender": request.gender,
                "Customer Type": request.customer_type,
                "Type of Travel": request.type_of_travel,
                "Class": request.flight_class,
            }
        ],
        columns=EXPECTED_FEATURES,
    )
    prediction = max(0.0, float(model.predict(model_input)[0]))
    return PredictionResponse(
        predicted_arrival_delay_minutes=round(prediction, 2),
        prediction_stage="after-departure",
        advisory=(
            "This is an educational estimate, not an operational guarantee. "
            "The model's held-out test MAE was approximately 6 minutes."
        ),
    )
