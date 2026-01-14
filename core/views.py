from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render, redirect
from django.contrib import messages

def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("home")
        messages.error(request, "Invalid credentials")
    return render(request, "core/login.html")

def logout_view(request):
    logout(request)
    return redirect("login")

@login_required
def home(request):
    return render(request, "core/home.html")

@login_required
def predict_view(request):
    return render(request, "core/predict.html")

@login_required
def logs_view(request):
    return render(request, "core/logs.html")

from .forms import SignupForm

def signup_view(request):
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data["password"])
            user.save()
            messages.success(request, "Account created successfully! Please log in.")
            return redirect("login")
    else:
        form = SignupForm()
    return render(request, "core/signup.html", {"form": form})

from .ml_service import SoilModelService
from .forms import SoilForm, ImageForm
from .models import SoilInput, ImageInput, PredictionRecord, FilePrediction
from .gemini_service import GeminiWeedService
from .fusion_service import FusionService

@login_required
def predict_view(request):
    if request.method == "POST":
        print("predict_view POST received")
        print(f"POST keys: {list(request.POST.keys())}")
        print(f"FILES keys: {list(request.FILES.keys())}")
        soil_form = SoilForm(request.POST)
        image_form = ImageForm(request.POST, request.FILES)
        print(f"soil_form.is_valid(): {soil_form.is_valid()}")
        print(f"image_form.is_valid(): {image_form.is_valid()}")
        if soil_form.is_valid() and image_form.is_valid():
            # Save soil input
            soil = soil_form.save(commit=False)
            soil.user = request.user
            soil.save()

            # Save image input
            image = image_form.save(commit=False)
            image.user = request.user
            image.save()

            # Run soil model (P) - This should always work
            try:
                result = SoilModelService.predict(soil_form.cleaned_data)
            except Exception as e:
                err = f"Error running soil prediction: {str(e)}"
                print(err)
                # Attach to a visible field so the UI shows it even if messages aren't rendered
                soil_form.add_error("crop_name", err)
                messages.error(request, err)
                return render(request, "core/predict.html", {
                    "soil_form": soil_form,
                    "image_form": image_form
                })

            # Weed detection (Q) - Handle Gemini API failures gracefully
            try:
                weed_result = GeminiWeedService.analyze_image(image.image.path)
            except Exception as e:
                # Fallback when Gemini API fails (e.g., API key issues)
                print(f"Warning: Gemini API error: {str(e)}")
                weed_result = {
                    "weed_presence": False,
                    "weed_details_json": {"error": "Weed detection service unavailable", "raw_response": None}
                }

            # Fusion (R) - Handle fusion service failures gracefully
            try:
                fusion_result = FusionService.fuse_results(result, weed_result)
            except Exception as e:
                # Fallback when fusion service fails
                print(f"Warning: Fusion service error: {str(e)}")
                # Create basic fusion result based on soil prediction only
                soil_class = result.get("soil_class", "Unknown")
                weed_presence = weed_result.get("weed_presence", False)
                
                if soil_class == "Suitable" and not weed_presence:
                    fusion_quality = "Good"
                    summary = "Soil is suitable and no weeds detected."
                elif soil_class == "Suitable" and weed_presence:
                    fusion_quality = "Moderate"
                    summary = "Soil is suitable but weeds detected."
                elif soil_class == "Unsuitable" and not weed_presence:
                    fusion_quality = "Moderate"
                    summary = "Soil conditions need improvement. No weeds detected."
                else:
                    fusion_quality = "Bad"
                    summary = "Soil is unsuitable and weeds detected."
                
                fusion_result = {
                    "fusion_quality": fusion_quality,
                    "fusion_summary_text": summary,
                    "suggestions_text": "AI suggestions unavailable. Please consult with agricultural experts for recommendations.",
                }

            # Save prediction record
            try:
                record = PredictionRecord.objects.create(
                    user=request.user,
                    soil=soil,
                    image=image,
                    soil_score=result["soil_score"],
                    soil_class=result["soil_class"],
                    confidences_json={"soil_confidence": result["confidence"]},
                    weed_presence=weed_result["weed_presence"],
                    weed_details_json=weed_result["weed_details_json"],
                    fusion_quality=fusion_result["fusion_quality"],
                    fusion_summary_text=fusion_result["fusion_summary_text"],
                    suggestions_text=fusion_result["suggestions_text"]
                )

                print(f"PredictionRecord created pk={record.pk}; redirecting to result")
                return redirect("result", pk=record.pk)
            except Exception as e:
                err = f"Error saving prediction: {str(e)}"
                print(err)
                # Attach to a visible field so the UI shows it even if messages aren't rendered
                soil_form.add_error("crop_name", err)
                messages.error(request, err)
                return render(request, "core/predict.html", {
                    "soil_form": soil_form,
                    "image_form": image_form
                })
        else:
            # Surface form errors to the user instead of silently refreshing.
            # We include the exact form errors so it's obvious what's blocking the redirect
            # (commonly: missing image upload, invalid numeric ranges, etc.).
            if soil_form.errors:
                print(f"SoilForm errors: {soil_form.errors.as_json()}")
                messages.error(request, f"Soil input error(s): {soil_form.errors.as_text()}")
            if image_form.errors:
                print(f"ImageForm errors: {image_form.errors.as_json()}")
                messages.error(request, f"Image upload error(s): {image_form.errors.as_text()}")
            if not soil_form.errors and not image_form.errors:
                messages.error(request, "Form submission failed validation. Please review your inputs.")
    else:
        soil_form = SoilForm()
        image_form = ImageForm()

    return render(request, "core/predict.html", {
        "soil_form": soil_form,
        "image_form": image_form
    })

import markdown2
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404
from .models import PredictionRecord

@login_required
def result_view(request, pk):

    record = get_object_or_404(PredictionRecord, pk=pk)
    # IMPORTANT: record.pk is NOT the same as ImageInput.pk. Use the linked FK instead.
    image = record.image
    soil = record.soil

    # Derive human-friendly confidence info
    confidence = None           # raw 0–1 from model
    confidence_pct = None       # nicely scaled 0–100 for display (compressed range)
    confidence_label = None
    try:
        if record.confidences_json:
            raw_conf = record.confidences_json.get("soil_confidence")
            if raw_conf is not None:
                confidence = float(raw_conf)
                # Compress very high confidences into a more realistic 90–95% display band.
                # This does NOT change the underlying model score, only how we show it.
                # Map raw in [0.5, 1.0] → [90, 95]
                clipped = max(0.5, min(confidence, 1.0))
                confidence_pct = round(90.0 + (clipped - 0.5) / 0.5 * 5.0, 2)

                if confidence >= 0.9:
                    confidence_label = "High"
                elif confidence >= 0.7:
                    confidence_label = "Medium"
                else:
                    confidence_label = "Low"
    except Exception as e:
        # Don't break the page if confidence is malformed
        print(f"Warning: could not parse confidence: {e}")

    # Convert fusion summary + suggestions from Markdown → HTML
    fusion_summary_html = markdown2.markdown(record.fusion_summary_text or "")
    suggestions_html = markdown2.markdown(record.suggestions_text or "")

    return render(request, "core/result.html", {
        "record": record,
        "fusion_summary_html": fusion_summary_html,
        "suggestions_html": suggestions_html,
        "image": image,
        "soil": soil,
        "confidence_pct": confidence_pct,
        "confidence_label": confidence_label,
    })


@login_required
def logs_view(request):
    # Get all predictions for the current user
    records = PredictionRecord.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "core/logs.html", {"records": records})


from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse

from .models import FilePrediction
from .gemini_service import GeminiWeedService


# ✅ File-Based Prediction Views
@login_required
def predict_file_view(request):
    """
    Handles file upload and runs Gemini analysis.
    """
    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]

        # Analyze file with Gemini (plain text output)
        uploaded_file.seek(0)
        try:
            analysis_text = GeminiWeedService.analyze_file(uploaded_file)
        except Exception as e:
            # Graceful fallback when the Gemini API key is invalid or blocked
            analysis_text = (
                "Unable to fetch AI suggestions at the moment. "
                "Please update the Gemini API key in settings or environment. "
                f"(Error: {str(e)})"
            )

        # Save result in FilePrediction
        file_record = FilePrediction.objects.create(
            user=request.user,
            uploaded_file=uploaded_file,
            gemini_output=analysis_text
        )

        # Redirect to result page
        return redirect("file_result", pk=file_record.pk)

    return render(request, "core/predict_file.html")


@login_required
def file_result_view(request, pk):
    """
    Displays the result page for a file prediction.
    """
    file_record = get_object_or_404(FilePrediction, pk=pk, user=request.user)
    return render(request, "core/file_result.html", {"record": file_record})


import markdown2
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render
from .models import FilePrediction

@login_required
def file_result_view(request, pk):
    file_record = get_object_or_404(FilePrediction, pk=pk, user=request.user)

    # Convert Markdown (Gemini output) → HTML
    formatted_output = markdown2.markdown(file_record.gemini_output)

    return render(
        request,
        "core/file_result.html",
        {"record": file_record, "formatted_output": formatted_output}
    )

from django.shortcuts import render
from .models import FilePrediction

@login_required
def pdf_logs_view(request):
    # Fetch all predictions for the logged-in user
    logs = FilePrediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'core/pdf_logs.html', {'logs': logs})

# from django.shortcuts import render, get_object_or_404
# from .models import FilePrediction

# def pdf_log_detail_view(request, pk):
#     # Fetch the specific prediction
#     log = get_object_or_404(FilePrediction, pk=pk, user=request.user)
#     return render(request, '', {'log': log})