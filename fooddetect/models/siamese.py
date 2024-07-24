import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.ImageOps
import numpy as np

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Function to load the model
def load_model(model_path):
    model = SiameseNetwork()
    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess the images
def preprocess_image(image_path, should_invert=True):
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    image = Image.open(image_path).convert("L")
    if should_invert:
        image = PIL.ImageOps.invert(image)
    image = transform(image)
    return image.unsqueeze(0)

def compare_img(loaded_path, reference_path):

    model = load_model(BASE_DIR / 'models' / 'compare-reference.pth')

    image1 = preprocess_image(loaded_path)
    image2 = preprocess_image(reference_path)

    output1, output2 = model(image1, image2)

    cosine_similarity = F.cosine_similarity(output1, output2)
    return cosine_similarity.item()

