from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", UploadOcr.as_view(), name="home"),
    path('download/<str:fileID>/', DownloadView.as_view(), name="download"),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
