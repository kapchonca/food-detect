import os
import pickle
from django.shortcuts import render
from django.urls import reverse
from fooddetect.settings import MEDIA_URL
from detect.forms import UploadFileForm
from detect.models import Standard
from models.detect import handle_uploaded_file, create_food_objects

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            raw_path = handle_uploaded_file(form.cleaned_data['file'])
            image_path = os.path.join(MEDIA_URL, 'processed/predict', raw_path)
            classes = create_food_objects(raw_path)
            classes.sort(key=lambda x: x.confidence, reverse=True)
            request.session['classes'] = pickle.dumps(classes).decode('latin1')
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
    classes = pickle.loads(request.session.get('classes', '').encode('latin1'))
    current_class = next(filter(lambda x: x.class_number == class_id, classes))
    class_info = {
        'class_name': query.class_name,
        'temperature': query.temperature,
        'weight': query.weight,
        'image_url': query.image.url,
        'image_path' : image_rez,
        'similarity' : current_class.similarity
    }

    return render(request, 'detect/class_details.html', {'class_info': class_info})
