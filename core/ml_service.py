import joblib
import numpy as np
import os
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "model", "soil_model.joblib")
SCALER_PATH = os.path.join(settings.BASE_DIR, "model", "scaler.joblib")

class SoilModelService:
    _model = None
    _scaler = None

    @classmethod
    def load_model(cls):
        if cls._model is None:
            cls._model = joblib.load(MODEL_PATH)
        if cls._scaler is None and os.path.exists(SCALER_PATH):
            cls._scaler = joblib.load(SCALER_PATH)

    @classmethod
    def predict(cls, soil_data: dict):
        """
        soil_data: dictionary with keys like ph, n, p, k, ec, organic_matter, etc.
        """
        cls.load_model()

        # Extract features in the same order used during training
        features = [
            soil_data.get("ph", 0),
            soil_data.get("n", 0),
            soil_data.get("p", 0),
            soil_data.get("k", 0),
            soil_data.get("ec", 0),
            soil_data.get("organic_matter", 0),
            soil_data.get("moisture", 0),
            soil_data.get("fe", 0),
            soil_data.get("zn", 0),
            soil_data.get("mn", 0),
            soil_data.get("cu", 0),
            soil_data.get("b", 0),
        ]

        X = np.array(features).reshape(1, -1)

        # Scale features
        if cls._scaler:
            X = cls._scaler.transform(X)

        # Predict
        prediction = cls._model.predict(X)[0]
        confidence = None
        if hasattr(cls._model, "predict_proba"):
            confidence = float(max(cls._model.predict_proba(X)[0]))

        soil_class = "Suitable" if prediction == 1 else "Unsuitable"

        return {
            "soil_score": float(prediction),
            "soil_class": soil_class,
            "confidence": confidence,
        }