import os
from django.shortcuts import render
from fooddetect.settings import MEDIA_ROOT, MEDIA_URL
from detect.forms import UploadFileForm
from models.siamese import compare_img
from models.detect import handle_uploaded_file, extract_classes

# Create your views here.
    
def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            raw_path = 'processed/predict/' + handle_uploaded_file(form.cleaned_data['file'])
            image_path = os.path.join(MEDIA_URL, raw_path)
            image_path_model = MEDIA_ROOT / raw_path
            similarity = compare_img(image_path_model, '/home/kapchonka/coding/dscs/food-detect/fooddetect/media/standard/11063_jpg.rf.jpg')
            classes = extract_classes(form.cleaned_data['file'].name)
    else:
        form = UploadFileForm()
        image_path = None
        classes = None
        
    return render(request, 'detect/index.html', {'form': form, 'image_path': image_path, 'classes': classes})
