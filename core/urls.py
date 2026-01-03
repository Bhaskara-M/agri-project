from django.urls import path
from . import views

urlpatterns = [
    path("signup/", views.signup_view, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("", views.home, name="home"),
    path("predict/", views.predict_view, name="predict"),
    path("logs/", views.logs_view, name="logs"),
    path("result/<int:pk>/", views.result_view, name="result"), 
]
