import os
from fooddetect.settings import BASE_DIR, MEDIA_ROOT
from ultralytics import YOLO

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
    model(MEDIA_ROOT / f'uploads/{file_name}', save=True, save_txt=True, project=MEDIA_ROOT / 'processed', exist_ok=True)

def extract_classes(file_name):
    file_extension = file_name.rfind('.')
    file_name = file_name[:file_extension] + '.txt'
    with open(BASE_DIR / 'models' / 'labels.txt') as file_labels:
        labels = file_labels.read().split('\n')

    if not(os.path.exists(MEDIA_ROOT / 'processed' / 'predict' / 'labels' / file_name)):
        return {'non-food': ['0', '0', '0', '0']}
    
    with open(MEDIA_ROOT / 'processed' / 'predict' / 'labels' / file_name) as file_predictions:
        classes_raw = file_predictions.read().split('\n')
    classes_list = [list(i.split(' ')) for i in classes_raw]
    classes_dict = { i[0] : i[1:] for i in classes_list if i[0]}
    classes = {labels[int(i)] : classes_dict[i] for i in classes_dict}

    return classes