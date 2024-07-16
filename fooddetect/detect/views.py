import os
from django.shortcuts import render
from fooddetect.settings import BASE_DIR, MEDIA_ROOT, MEDIA_URL
from detect.forms import UploadFileForm
from ultralytics import YOLO

# Create your views here.

def handle_uploaded_file(f):
    upload_dir = MEDIA_ROOT / 'uploads/'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    with open(os.path.join(upload_dir, f.name), "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    
    process_image(f.name)
    return f.name

def process_image(file_name):
    upload_dir = MEDIA_ROOT / 'processed/'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    model = YOLO(BASE_DIR / 'models/detect.pt')
    model(MEDIA_ROOT / f'uploads/{file_name}', save=True, project=MEDIA_ROOT / 'processed', exist_ok=True)
    
def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            image_path = MEDIA_URL + 'processed/predict/' + handle_uploaded_file(form.cleaned_data['file'])
    else:
        form = UploadFileForm()
        image_path = None
    print(image_path)
    return render(request, 'detect/index.html', {'form': form, 'image_path': image_path})
