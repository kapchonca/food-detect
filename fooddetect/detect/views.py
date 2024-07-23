import os
from django.shortcuts import render
from fooddetect.settings import MEDIA_ROOT, MEDIA_URL, BASE_DIR
from detect.forms import UploadFileForm
from models.detect import handle_uploaded_file, create_food_objects, FoodObject
    
def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            raw_path = handle_uploaded_file(form.cleaned_data['file'])
            image_path = os.path.join(MEDIA_URL , 'processed/predict', raw_path)
            classes = create_food_objects(raw_path)

    else:
        form = UploadFileForm()
        image_path = None
        classes = None
        
    return render(request, 'detect/index.html', {'form': form, 'image_path': image_path, 'classes': classes})
