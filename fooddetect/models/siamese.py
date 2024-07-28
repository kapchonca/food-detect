import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from fooddetect.settings import BASE_DIR
from detect.models import Standard


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image


def get_features(image_tensor: torch.Tensor, model: torch.nn.Module):
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()


def load_model(model_path: str) -> torch.nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 89)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def calculate_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    euclidean_distance = np.linalg.norm(features1 - features2)
    return round(max(0, 100 - euclidean_distance), 2)


def compare_images(loaded_path: str, reference_name: str) -> float:

    model = load_model(BASE_DIR / "models" / "compare-reference.pth")

    image1_tensor = preprocess_image(loaded_path)
    features1 = get_features(image1_tensor, model)

    features2_query = Standard.objects.get(class_name=reference_name)
    features2 = np.array(features2_query.embedding)

    similarity_percentage = calculate_similarity(features1, features2)
    return similarity_percentage
