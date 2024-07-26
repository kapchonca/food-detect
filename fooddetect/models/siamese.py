import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from fooddetect.settings import BASE_DIR
from detect.models import Standard

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def get_features(image_tensor, model):
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()

def compare_img(loaded_path, reference_name):

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 89)
    model.load_state_dict(torch.load(BASE_DIR / 'models' / 'compare-reference.pth'))
    model.eval()

    image1_tensor = preprocess_image(loaded_path)
    features1 = get_features(image1_tensor, model)

    features2_query = Standard.objects.get(class_name=reference_name)
    features2 = np.array(features2_query.embedding)

    euclidean_distance = np.linalg.norm(features1 - features2)
    
    similarity_percentage = max(0, 100 - euclidean_distance)
    return round(similarity_percentage, 2)
