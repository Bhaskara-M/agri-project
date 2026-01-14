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
        """Load model and scaler if not already loaded."""
        if cls._model is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
            try:
                cls._model = joblib.load(MODEL_PATH)
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
        
        if cls._scaler is None and os.path.exists(SCALER_PATH):
            try:
                cls._scaler = joblib.load(SCALER_PATH)
            except Exception as e:
                print(f"Warning: Could not load scaler: {str(e)}. Proceeding without scaling.")
                cls._scaler = None

    @classmethod
    def _convert_to_float(cls, value, default=0.0):
        """Convert value to float, handling None and string inputs."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def predict(cls, soil_data: dict):
        """
        Predict soil quality from soil data dictionary.
        
        Args:
            soil_data: dictionary with keys like ph, n, p, k, ec, organic_matter, etc.
                      Values can be float, int, string, or None.
        
        Returns:
            dict with keys: soil_score (0 or 1), soil_class ("Suitable" or "Unsuitable"), 
                           confidence (0-1 float or None)
        
        Raises:
            Exception: If model cannot be loaded or prediction fails
        """
        cls.load_model()

        # Extract features in the same order used during training
        # Handle None values and convert to float
        features = [
            cls._convert_to_float(soil_data.get("ph"), 0.0),
            cls._convert_to_float(soil_data.get("n"), 0.0),
            cls._convert_to_float(soil_data.get("p"), 0.0),
            cls._convert_to_float(soil_data.get("k"), 0.0),
            cls._convert_to_float(soil_data.get("ec"), 0.0),
            cls._convert_to_float(soil_data.get("organic_matter"), 0.0),
            cls._convert_to_float(soil_data.get("moisture"), 0.0),
            cls._convert_to_float(soil_data.get("fe"), 0.0),
            cls._convert_to_float(soil_data.get("zn"), 0.0),
            cls._convert_to_float(soil_data.get("mn"), 0.0),
            cls._convert_to_float(soil_data.get("cu"), 0.0),
            cls._convert_to_float(soil_data.get("b"), 0.0),
        ]

        # Convert to numpy array and ensure correct shape
        try:
            X = np.array(features, dtype=np.float64).reshape(1, -1)
            
            # Validate feature array
            if X.shape[1] != 12:
                raise ValueError(f"Expected 12 features, got {X.shape[1]}")
            
            # Check for NaN or Inf values
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                # Replace NaN/Inf with 0
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            raise Exception(f"Error preparing features: {str(e)}")

        # Scale features if scaler is available
        try:
            if cls._scaler:
                X = cls._scaler.transform(X)
        except Exception as e:
            raise Exception(f"Error scaling features: {str(e)}")

        # Predict
        try:
            prediction = cls._model.predict(X)[0]
            
            # Ensure prediction is 0 or 1
            prediction = int(prediction)
            if prediction not in [0, 1]:
                prediction = 1 if prediction > 0 else 0
            
            # Get confidence score if available
            confidence = None
            if hasattr(cls._model, "predict_proba"):
                try:
                    proba = cls._model.predict_proba(X)[0]
                    confidence = float(max(proba))
                except Exception as e:
                    print(f"Warning: Could not get prediction probability: {str(e)}")
            
            soil_class = "Suitable" if prediction == 1 else "Unsuitable"

            # Log prediction details for debugging (can be removed in production)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"ML Prediction - Input: {soil_data}, Output: {prediction} ({soil_class}), Confidence: {confidence}")

            return {
                "soil_score": float(prediction),
                "soil_class": soil_class,
                "confidence": confidence,
            }
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")