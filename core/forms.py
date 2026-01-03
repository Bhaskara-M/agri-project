from django import forms
from django.contrib.auth.models import User

class SignupForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ["username", "email", "password"]

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")
        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match")
        return cleaned_data


from .models import SoilInput, ImageInput

class SoilForm(forms.ModelForm):
    class Meta:
        model = SoilInput
        fields = ["crop_name","ph","n","p","k","ec","organic_matter","moisture","texture","fe","zn","mn","cu","b"]

    def clean_ph(self):
        ph = self.cleaned_data["ph"]
        if not (3.0 <= ph <= 10.0):
            raise forms.ValidationError("pH must be between 3.0 and 10.0")
        return ph

class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageInput
        fields = ["image","capture_date","notes"]