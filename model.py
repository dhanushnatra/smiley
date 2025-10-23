
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
from torchvision import transforms
from PIL import Image


#specify transformations

transform  = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# rebuild Model

class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    


# load classses [ angry , sad , ....]

with open('classes.pkl', 'rb') as f:
    exp_classes = pickle.load(f)
    
    
# load model onto device and use cuda if its available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceClassifier(len(exp_classes))
state = torch.load('face_classifier.pth', map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

def predict(image_arr):
    img: Image.Image = Image.fromarray(image_arr)  # type: ignore
    img: torch.Tensor = transform(img)
    with torch.no_grad():
        outputs = model(img.unsqueeze(0).to(device))
        _, predicted = torch.max(outputs, 1)
    return exp_classes[predicted.item()]


def predict_image_file(image_path):
    image = Image.open(image_path).convert('RGB')
    image = numpy.array(image)
    return predict(image)





