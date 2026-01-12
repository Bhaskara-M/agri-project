from google import genai
from django.conf import settings

client = genai.Client(api_key="AIzaSyDrbtzb0cwkBJVEUrPv2CWpmm1bkpOjroc")  # safer: keep key in settings

class FusionService:
    @staticmethod
    def fuse_results(soil_result: dict, weed_result: dict):
        """
        soil_result: dict from SoilModelService.predict
        weed_result: dict from GeminiWeedService.analyze_image
        Returns dict with fusion_quality, summary, suggestions (from Gemini).
        """

        soil_class = soil_result.get("soil_class")
        weed_presence = weed_result.get("weed_presence")

        # Fusion quality + summary logic
        if soil_class == "Suitable" and not weed_presence:
            fusion_quality = "Good"
            summary = "Soil is suitable and no weeds detected."
        elif soil_class == "Suitable" and weed_presence:
            fusion_quality = "Moderate"
            summary = "Soil is suitable but weeds detected."
        elif soil_class == "Unsuitable" and not weed_presence:
            fusion_quality = "Good"
            summary = "Soil is unsuitable despite no weeds."
        else:  # Unsuitable + Weeds
            fusion_quality = "Bad"
            summary = "Soil is unsuitable and weeds detected."

        # 🔹 Ask Gemini for suggestions
        prompt = (
            f"Soil class: {soil_class}. "
            f"Weed presence: {weed_presence}. "
            "Talk about above conditions"
            "Based on these conditions, provide practical farming suggestions "
            "Give brief concise, most useful suggestions or solutions relating to the given conditions"
            "Use bold labels and bullet points. "
        )

        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
            )
            # Extract text safely
            if hasattr(response, "text"):
                suggestions = response.text.strip()
            elif hasattr(response, "candidates"):
                suggestions = response.candidates[0].content.parts[0].text.strip()
            else:
                suggestions = "No AI suggestions available."
        except Exception as e:
            suggestions = "Unable to fetch AI suggestions at the moment. Please rely on standard agronomy practices."

        return {
            "fusion_quality": fusion_quality,
            "fusion_summary_text": summary,
            "suggestions_text": suggestions,
        }