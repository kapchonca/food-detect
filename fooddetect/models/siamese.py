import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from fooddetect.settings import BASE_DIR
from detect.models import Standard


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocesses the input image for the siamese model.

    Args:
        image_path: The path to the image file.

    Returns:
        A tensor representing the preprocessed image.
    """

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
    """
    Extracts features from the image tensor using the given model.

    Args:
        image_tensor: A tensor representing the preprocessed image.
        model: The model used to extract features.

    Returns:
        A numpy array containing the extracted features.
    """

    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()


def load_model(model_path: str) -> torch.nn.Module:
    """
    Loads the model with custom weights.

    Args:
        model_path: The path to the model weights file.

    Returns:
        The loaded model.
    """

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 89)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def calculate_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Calculates the similarity between two sets of features using Euclidean distance.

    Args:
        features1: The first set of features.
        features2: The second set of features.

    Returns:
        The similarity percentage between the two feature sets.
    """

    euclidean_distance = np.linalg.norm(features1 - features2)
    return round(max(0, 100 - euclidean_distance), 2)


def compare_images(loaded_path: str, reference_name: str) -> float:
    """
    Compares the extracted features of a loaded image with the features of a reference image.

    Args:
        loaded_path: The path to the loaded image file.
        reference_name: The name of the reference class.

    Returns:
        The similarity percentage between the loaded image and the reference image.
    """

    model = load_model(BASE_DIR / "models" / "compare-reference.pth")

    image1_tensor = preprocess_image(loaded_path)
    features1 = get_features(image1_tensor, model)

    features2_query = Standard.objects.get(class_name=reference_name)
    features2 = np.array(features2_query.embedding)

    similarity_percentage = calculate_similarity(features1, features2)
    return similarity_percentage
