from google import genai
from django.conf import settings

client = genai.Client(api_key="AIzaSyACklqM2e-vUSvium1p5FXQ__mMGh59aSg")

class FusionService:
    @staticmethod
    def fuse_results(soil_result: dict, weed_result: dict):
        """
        soil_result: dict from SoilModelService.predict
        weed_result: dict from GeminiWeedService.analyze_image
        Returns dict with fusion_quality, summary, suggestions (from Gemini).
        """

        soil_class = soil_result["soil_class"]
        weed_presence = weed_result["weed_presence"]

        # Fusion quality + summary logic (still deterministic)
        if soil_class == "Suitable" and not weed_presence:
            fusion_quality = "Good"
            summary = "Soil is suitable and no weeds detected."
        elif soil_class == "Suitable" and weed_presence:
            fusion_quality = "Moderate"
            summary = "Soil is suitable but weeds detected."
        elif soil_class == "Unsuitable" and not weed_presence:
            fusion_quality = "Poor"
            summary = "Soil is unsuitable despite no weeds."
        else:  # Unsuitable + Weeds
            fusion_quality = "Bad"
            summary = "Soil is unsuitable and weeds detected."

        # 🔹 Ask Gemini for suggestions (single paragraph, 2–3 sentences)
        prompt = (
            f"Soil class: {soil_class}. "
            f"Weed presence: {weed_presence}. "
            "Based on these conditions, provide practical farming suggestions "
            "in a single paragraph of 2–3 sentences."
        )

        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            suggestions = response.text.strip()
            suggestions = suggestions.replace("*", "")
            suggestions = suggestions.replace("#", "")
        except Exception as e:
            # Fallback if Gemini fails
            suggestions = "Unable to fetch AI suggestions at the moment. Please rely on standard agronomy practices."

        return {
            "fusion_quality": fusion_quality,
            "fusion_summary_text": summary,
            "suggestions_text": suggestions,
        }