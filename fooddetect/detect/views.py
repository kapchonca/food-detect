import os
from django.shortcuts import render
from fooddetect.settings import MEDIA_ROOT, MEDIA_URL
from detect.forms import UploadFileForm
from models.siamese import compare_img
from models.detect import handle_uploaded_file, extract_classes
from detect.models import Standard

# Create your views here.
    
def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            raw_path = 'processed/predict/' + handle_uploaded_file(form.cleaned_data['file'])
            image_path = os.path.join(MEDIA_URL, raw_path)
            image_path_model = MEDIA_ROOT / raw_path
            classes = extract_classes(form.cleaned_data['file'].name)

            if classes:
                query_set = Standard.objects.filter(class_number__in=classes.keys())
                for query in query_set:
                    print(query.image)
                        
                similarity = compare_img(image_path_model, os.path.join(MEDIA_ROOT, 'standard/11020_jpg.rf.jpg'))
            else:
                classes = {'non-food' : '0'}

    else:
        form = UploadFileForm()
        image_path = None
        classes = None
        
    return render(request, 'detect/index.html', {'form': form, 'image_path': image_path, 'classes': classes})
