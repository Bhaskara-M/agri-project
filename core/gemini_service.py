import google.genai as genai   # ✅ correct import if you installed `google-genai`
from django.conf import settings

# Configure Gemini API
client = genai.Client(api_key="AIzaSyDrbtzb0cwkBJVEUrPv2CWpmm1bkpOjroc")  # ⚠️ Replace with settings.GEMINI_API_KEY if stored securely

class GeminiWeedService:
    @staticmethod
    def analyze_image(image_path: str):
        """
        Analyze crop image for weed detection using Gemini.
        Returns dict with weed_presence (True/False) and weed_details_json.
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = (
            "Analyze this agricultural file and provide results in Markdown format. "
            "Use bold labels and bullet points. Include:\n"
            "- Weed Presence: Present or Absent\n"
            "- Soil Suitability: Suitable or Not Suitable\n"
            "- Suggestions: Provide practical recommendations as bullet points"
        )

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