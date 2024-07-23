import os
from ultralytics import YOLO
from fooddetect.settings import BASE_DIR, MEDIA_ROOT

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
    detections = model(MEDIA_ROOT / f'uploads/{file_name}', save=True, project=MEDIA_ROOT / 'processed', exist_ok=True, conf=0.1)
    process_detections(detections, file_name)

def process_detections(detections, file_name):
    top_detections = {}

    for detection in detections:
        for box in detection.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)

            if class_id not in top_detections:
                top_detections[class_id] = 0

            top_detections[class_id] = max(confidence, top_detections[class_id])
            
    file_name = img_to_txt_filename(file_name)

    with open(MEDIA_ROOT / 'processed/predict/labels' / file_name, 'w') as f:
        for class_id, confidence in top_detections.items():
            f.write(f"{class_id} {confidence}")
            f.write("\n")

def extract_classes(file_name):
    file_name = img_to_txt_filename(file_name)
    
    with open(MEDIA_ROOT / 'processed' / 'predict' / 'labels' / file_name) as file_predictions:
        classes_raw = file_predictions.read().split('\n')
    classes_list = [list(i.split(' ')) for i in classes_raw]
    classes = { i[0] : i[1] for i in classes_list if i[0]}

    return classes