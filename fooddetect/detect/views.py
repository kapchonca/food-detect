import os
from django.shortcuts import render
from fooddetect.settings import MEDIA_URL, BASE_DIR
from detect.forms import UploadFileForm
from detect.models import Standard
from models.detect import save_uploaded_file, extract_classes
from models.siamese import compare_images


def index(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file_name = save_uploaded_file(form.cleaned_data["file"])
            image_path = os.path.join(MEDIA_URL, "processed/predict", file_name)
            classes = extract_classes(file_name)
            classes.sort(key=lambda x: x.confidence, reverse=True)
            request.session["image_path"] = image_path

            context = {"image_path": image_path, "classes": classes}
            return render(request, "detect/results.html", context=context)
    else:
        context = {"form": UploadFileForm(), "image_path": None, "classes": None}

    return render(request, "detect/index.html", context=context)


def all_classes(request):
    all_classes_query = Standard.objects.all()
    return render(
        request, "detect/all_classes.html", {"all_classes": all_classes_query}
    )


def class_details(request, class_id):
    if class_id is None:
        return render(
            request, "detect/all_classes.html", {"all_classes": Standard.objects.all()}
        )

    query = Standard.objects.get(class_number=class_id)
    image_path = request.session.get("image_path", "")

    similarity = compare_images(
        BASE_DIR / image_path[1:], query.class_name
    )  # [1:] removes slash in front of the relative path

    class_info = {
        "class_name": query.class_name,
        "temperature": query.temperature,
        "weight": query.weight,
        "image_url": query.image.url,
        "image_path": image_path,
        "similarity": similarity,
    }

    return render(request, "detect/class_details.html", {"class_info": class_info})
