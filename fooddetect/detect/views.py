import os
from django.shortcuts import render
from fooddetect.settings import MEDIA_URL, BASE_DIR
from detect.forms import UploadFileForm
from detect.models import Standard
from models.detect import handle_uploaded_file, create_food_objects
from models.siamese import compare_img

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            raw_path = handle_uploaded_file(form.cleaned_data['file'])
            image_path = os.path.join(MEDIA_URL, 'processed/predict', raw_path)
            classes = create_food_objects(raw_path)
            classes.sort(key=lambda x: x.confidence, reverse=True)
            request.session['image_path'] = image_path
            return render(request, 'detect/results.html', {'image_path': image_path, 'classes': classes})
    else:
        form = UploadFileForm()
        image_path = None
        classes = None

    return render(request, 'detect/index.html', {'form': form, 'image_path': image_path, 'classes': classes})

def all_classes(request):
    all_classes = Standard.objects.all()
    return render(request, 'detect/all_classes.html', {'all_classes': all_classes})

def class_details(request, class_id):
    if class_id is None:
        return render(request, 'detect/all_classes.html', {'all_classes': Standard.objects.all()})

    query = Standard.objects.get(class_number=class_id)
    image_rez = request.session.get('image_path', '')
    
    similarity = compare_img(BASE_DIR / image_rez[1:], query.class_name) # [1:] removes slash in front of the relative path

    class_info = {
        'class_name': query.class_name,
        'temperature': query.temperature,
        'weight': query.weight,
        'image_url': query.image.url,
        'image_path': image_rez,
        'similarity': similarity
    }

    return render(request, 'detect/class_details.html', {'class_info': class_info})
