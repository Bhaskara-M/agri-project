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
from .models import SoilInput, ImageInput, PredictionRecord
from .gemini_service import GeminiWeedService
from .fusion_service import FusionService



@login_required
def predict_view(request):
    if request.method == "POST":
        soil_form = SoilForm(request.POST)
        image_form = ImageForm(request.POST, request.FILES)
        if soil_form.is_valid() and image_form.is_valid():
            # Save soil input
            soil = soil_form.save(commit=False)
            soil.user = request.user
            soil.save()

            # Save image input
            image = image_form.save(commit=False)
            image.user = request.user
            image.save()

            # Run soil model (P)
            result = SoilModelService.predict(soil_form.cleaned_data)

            # Weed detection (Q)
            weed_result = GeminiWeedService.analyze_image(image.image.path)

            # Fusion (R)
            fusion_result = FusionService.fuse_results(result, weed_result)

            # Save prediction record
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

            return redirect("result", pk=record.pk)
    else:
        soil_form = SoilForm()
        image_form = ImageForm()

    return render(request, "core/predict.html", {
        "soil_form": soil_form,
        "image_form": image_form
    })

@login_required
def result_view(request, pk):
    record = get_object_or_404(PredictionRecord, pk=pk)
    suggestions_list = record.suggestions if hasattr(record, "suggestions") else []
    return render(request, "core/result.html", {"record": record, "suggestions_list": suggestions_list
})


@login_required
def logs_view(request):
    # Get all predictions for the current user
    records = PredictionRecord.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "core/logs.html", {"records": records})


