from django.urls import path
from detect import views

urlpatterns = [
    path('home/', views.index),
    path('details/<int:class_id>/', views.class_details, name='class_details'),
    path('all_classes/', views.all_classes, name='all_classes'),
]
