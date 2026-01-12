from django.conf import settings
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("signup/", views.signup_view, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("", views.home, name="home"),
    path("predict/", views.predict_view, name="predict"),
    path("logs/", views.logs_view, name="logs"),
    path("result/<int:pk>/", views.result_view, name="result"),

    # ✅ New routes for File-Based Prediction
    path("predict/file/", views.predict_file_view, name="predict_file"),
    path("result/file/<int:pk>/", views.file_result_view, name="file_result"),
    
    path('logs/pdf/', views.pdf_logs_view, name='pdf_logs'),
    path('logs/pdf/<int:pk>/', views.file_result_view, name='pdf_log_detail'),

]


if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
