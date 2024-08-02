from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("all_classes/", views.all_classes, name="all_classes"),
    path("details/<int:class_id>/", views.class_details, name="class_details"),
]
