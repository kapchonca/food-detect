import os
from django.shortcuts import render
from fooddetect.settings import BASE_DIR
from detect.forms import UploadFileForm

# Create your views here.

def handle_uploaded_file(f):
    upload_dir = BASE_DIR / 'uploads/'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    with open(os.path.join(upload_dir, f.name), "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(form.cleaned_data['file'])
    else:
        form = UploadFileForm()
    return render(request, 'detect/index.html', {'form': form})
