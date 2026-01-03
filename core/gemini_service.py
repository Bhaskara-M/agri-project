import os
import google.genai as genai   # ✅ correct import if you installed `google-genai`
from django.conf import settings

# Configure Gemini API
client = genai.Client(api_key="AIzaSyACklqM2e-vUSvium1p5FXQ__mMGh59aSg")

class GeminiWeedService:
    @staticmethod
    def analyze_image(image_path: str):
        """
        Analyze crop image for weed detection using Gemini.
        Returns dict with weed_presence (True/False) and weed_details_json.
        """

        # Load image
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Prompt Gemini to return only one word: "Present" or "Absent"
        prompt = (
            "Analyze this crop image and respond with a single word only: "
            "'Present' if weeds are detected, or 'Absent' if no weeds are detected."
        )

        # Call Gemini Vision model
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
                    ],
                }
            ],
        )

        # Normalize response
        result_text = (response.text or "").strip().lower()

        if "present" in result_text:
            weed_presence = True
        elif "absent" in result_text:
            weed_presence = False
        else:
            # Fallback: keyword check
            weed_presence = "weed" in result_text

        weed_details = {"raw_response": response.text}

        return {
            "weed_presence": weed_presence,
            "weed_details_json": weed_details,
        }
