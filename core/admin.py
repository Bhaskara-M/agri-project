from django.contrib import admin
from .models import SoilInput, ImageInput, PredictionRecord

admin.site.register(SoilInput)
admin.site.register(ImageInput)
admin.site.register(PredictionRecord)