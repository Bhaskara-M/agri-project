from django.db import models
from django.contrib.auth.models import User

class SoilInput(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    crop_name = models.CharField(max_length=100)
    ph = models.FloatField()
    n = models.FloatField(help_text="Nitrogen")
    p = models.FloatField(help_text="Phosphorus")
    k = models.FloatField(help_text="Potassium")
    ec = models.FloatField(null=True, blank=True, help_text="Electrical Conductivity")
    organic_matter = models.FloatField(null=True, blank=True)
    moisture = models.FloatField(null=True, blank=True)
    texture = models.CharField(max_length=50, null=True, blank=True)  # sandy, loamy, clayey
    fe = models.FloatField(null=True, blank=True)
    zn = models.FloatField(null=True, blank=True)
    mn = models.FloatField(null=True, blank=True)
    cu = models.FloatField(null=True, blank=True)
    b = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class ImageInput(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to="uploads/")
    capture_date = models.DateField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class PredictionRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    soil = models.ForeignKey(SoilInput, on_delete=models.CASCADE)
    image = models.ForeignKey(ImageInput, on_delete=models.CASCADE)
    soil_score = models.FloatField(null=True, blank=True)           # P
    soil_class = models.CharField(max_length=50, null=True, blank=True)
    weed_presence = models.BooleanField(default=False)              # Q
    weed_details_json = models.JSONField(null=True, blank=True)
    fusion_quality = models.CharField(max_length=20)                # good/bad
    fusion_summary_text = models.TextField()
    suggestions_text = models.TextField()
    confidences_json = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)