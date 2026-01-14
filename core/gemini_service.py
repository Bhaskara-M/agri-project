import google.genai as genai   # ✅ correct import if you installed `google-genai`
from django.conf import settings
import os

# Configure Gemini API.
# SECURITY: Do not hardcode API keys in source. Use env var or settings instead.
api_key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
if not api_key:
    # Defer failure to runtime with a clear error message (handled by callers).
    client = None
else:
    client = genai.Client(api_key=api_key)

class GeminiWeedService:
    @staticmethod
    def analyze_image(image_path: str):
        """
        Analyze crop image for weed detection using Gemini.
        Returns dict with weed_presence (True/False) and weed_details_json.
        
        Raises:
            Exception: If API call fails or image cannot be read
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            raise Exception(f"Error reading image file: {str(e)}")

        prompt = (
            "Analyze this agricultural file and provide results in Markdown format. "
            "Use bold labels and bullet points. Include:\n"
            "- Weed Presence: Present or Absent\n"
            "- Soil Suitability: Suitable or Not Suitable\n"
            "- Suggestions: Provide practical recommendations as bullet points"
        )

        if client is None:
            raise Exception("Gemini API key not configured. Set GEMINI_API_KEY in env or settings.")

        try:
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

            result_text = (response.text or "").strip().lower()

            if "present" in result_text:
                weed_presence = True
            elif "absent" in result_text:
                weed_presence = False
            else:
                weed_presence = "weed" in result_text

            weed_details = {"raw_response": response.text}

            return {
                "weed_presence": weed_presence,
                "weed_details_json": weed_details,
            }
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Gemini API error: {str(e)}")

    @staticmethod
    def analyze_file(file_obj):
        """
        Analyze an uploaded file (PDF or image) using Gemini.
        Returns well-formatted plain text with bold labels and bullet points.
        """
        file_bytes = file_obj.read()

        if file_obj.name.lower().endswith(".pdf"):
            mime_type = "application/pdf"
        else:
            mime_type = "image/jpeg"

        prompt = (
            "Analyze this agricultural file and provide results in a well-formatted way, "
            "like the Gemini app style. Use bold labels and bullet points. "
            "Include:\n"
            "- Weed Presence: Present or Absent\n"
            "- Soil Suitability: Suitable or Not Suitable\n"
            "- Suggestions: Provide practical recommendations as bullet points"
        )

        if client is None:
            raise Exception("Gemini API key not configured. Set GEMINI_API_KEY in env or settings.")

        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime_type, "data": file_bytes}},
                    ],
                }
            ],
        )

        # ✅ Return plain text directly (no dict)
        return (response.text or "").strip()