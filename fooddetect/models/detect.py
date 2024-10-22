import os
from dataclasses import dataclass
from typing import List
from ultralytics import YOLO
from fooddetect.settings import BASE_DIR, MEDIA_ROOT
from detect.models import Standard


@dataclass
class FoodObject:
    """
    Represents a detected food object.

    Attributes:
        class_name: The name of the food class.
        class_number: The numerical identifier of the food class.
        confidence: The confidence score of the detection.
    """

    class_name: str = "non-food"
    class_number: int = 0
    confidence: float = 0.0


def save_uploaded_file(file) -> str:
    """
    Saves an uploaded file to the uploads directory.

    Args:
        file: The file object to be saved.

    Returns:
        The name of the saved file.
    """

    upload_dir = MEDIA_ROOT / "uploads/"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    with open(os.path.join(upload_dir, file.name), "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return file.name


def process_image(file_name: str):
    """
    Processes the uploaded image using the YOLO model.

    Args:
        file_name: The name of the file to be processed.

    Returns:
        The result of the model prediction for the image.
    """

    processed_dir = MEDIA_ROOT / "processed/"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    model = YOLO(BASE_DIR / "models/detect.pt")
    results = model.predict(
        MEDIA_ROOT / f"uploads/{file_name}",
        save=True,
        project=processed_dir,
        exist_ok=True,
    )
    return results[0]


def extract_classes_dict(uploaded_path: str) -> dict:
    """
    Extracts the top detected classes and their confidence scores.

    Args:
        uploaded_path: The path to the uploaded image.

    Returns:
        A dictionary mapping class numbers to confidence scores.
    """

    result = process_image(uploaded_path)

    top_classes = result.probs.top5
    top_confidences = result.probs.top5conf
    classes_dict = {int(top_classes[i]): float(top_confidences[i]) for i in range(3)}

    return classes_dict


def extract_classes(uploaded_path: str) -> List[FoodObject]:
    """
    Extracts food objects from the uploaded image.

    Args:
        uploaded_path: The path to the uploaded image.

    Returns:
        A list of FoodObject instances representing detected food objects.
    """

    classes_dict = extract_classes_dict(uploaded_path)
    classes = []

    if classes_dict:
        query_set = Standard.objects.filter(class_number__in=classes_dict.keys())
        for query in query_set:
            food_object = FoodObject(
                class_name=query.class_name,
                class_number=query.class_number,
                confidence=classes_dict[query.class_number],
            )
            classes.append(food_object)
    else:
        classes = [FoodObject()]

    return classes
