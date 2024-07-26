import os
from ultralytics import YOLO
from fooddetect.settings import BASE_DIR, MEDIA_ROOT
from dataclasses import dataclass
from detect.models import Standard

@dataclass
class FoodObject:
    class_name: str = ''
    class_number: int = 0
    confidence: float = 0
    similarity: float = 0
    temperature: float = 0
    weight: float = 0
    image_path: str = ''

def handle_uploaded_file(f):
    upload_dir = MEDIA_ROOT / 'uploads/'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    with open(os.path.join(upload_dir, f.name), "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    
    process_image(f.name)
    return f.name

def img_to_txt_filename(file_name):
    file_extension = file_name.rfind('.')
    return file_name[:file_extension] + '.txt'

def process_image(file_name):
    upload_dir = MEDIA_ROOT / 'processed/'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    model = YOLO(BASE_DIR / 'models/detect.pt')
    results = model.predict(MEDIA_ROOT / f'uploads/{file_name}', save=True, project=MEDIA_ROOT / 'processed', exist_ok=True)
    return results[0]

def extract_classes_dict(uploaded_path):
    result = process_image(uploaded_path)

    class_numbers = result.probs.top5
    confidences = result.probs.top5conf
    classes_dict = {int(class_numbers[i]) : float(confidences[i].item()) for i in range(3)}

    return classes_dict

def create_food_objects(uploaded_path):
    classes_dict = extract_classes_dict(uploaded_path)
    classes = []
    
    if classes_dict:
        query_set = Standard.objects.filter(class_number__in=classes_dict.keys())
        for query in query_set:
            current_class = FoodObject()
            current_class.class_number = query.class_number
            current_class.confidence = classes_dict[query.class_number]
            current_class.class_name = query.class_name
            classes.append(current_class)    
    else:
        classes = [FoodObject(class_name='non-food')]

    return classes