import os
import random
import shutil
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from detect.models import Standard
from fooddetect.settings import BASE_DIR


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess the input image by resizing, normalizing, and converting it to a tensor.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
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
    Extract features from the image tensor using the specified model.

    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        model (torch.nn.Module): Pretrained model to extract features.

    Returns:
        numpy.ndarray: Extracted features as a numpy array.
    """
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that the specified directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def copy_image_to_media(image_path: str, target_directory: str) -> str:
    """
    Copy the image to the target directory, creating the directory if it doesn't exist.

    Args:
        image_path (str): Path to the source image.
        target_directory (str): Path to the target directory.

    Returns:
        str: Path to the copied image in the target directory.
    """
    ensure_directory_exists(target_directory)
    base_name = os.path.basename(image_path)
    target_path = os.path.join(target_directory, base_name)
    shutil.copy2(image_path, target_path)
    return target_path


def populate_database(root_folder: str) -> None:
    """
    Populate the database with images and their corresponding metadata.

    Args:
        root_folder (str): Root folder containing class subfolders with images.
    """
    media_folder = os.path.join("media", "standard")

    class_folders = [
        f
        for f in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, f))
    ]
    class_folders = sorted(class_folders)

    for class_number, class_folder in enumerate(class_folders, start=60):
        class_folder_path = os.path.join(root_folder, class_folder)
        images = [
            f
            for f in os.listdir(class_folder_path)
            if os.path.isfile(os.path.join(class_folder_path, f))
        ]

        if images:
            image_path = os.path.join(class_folder_path, images[0])

            new_image_path = copy_image_to_media(image_path, media_folder)
            new_image_path = new_image_path.replace("media/", "")

            temperature = random.randint(20, 60)
            weight = random.randint(100, 200)

            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, 89)
            model.load_state_dict(
                torch.load(
                    BASE_DIR / "models" / "compare-reference.pth",
                    map_location=torch.device("cpu"),
                )
            )
            model.eval()
            image_tensor = preprocess_image(image_path)
            features = get_features(image_tensor, model)
            embedding = features.tolist()

            Standard.objects.create(
                class_number=class_number,
                class_name=class_folder,
                temperature=temperature,
                weight=weight,
                image=new_image_path,
                embedding=embedding,
            )
            print(f"Added data for class {class_folder} with image {new_image_path}")


def run() -> None:
    """
    Run the process to populate the database with test data.
    """
    root_folder = BASE_DIR / "db-test-data" / "standards"
    populate_database(root_folder)
