import pickle
import torch
import cv2
import warnings
warnings.filterwarnings("ignore") 
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import time

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load the model architecture
model_loaded :nn.Sequential=  nn.Sequential(
    models.resnet18(pretrained=True),
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Linear(512, 8),
    nn.LogSoftmax(dim=1)
)
# Setup device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loaded.to(device)
print(f"Model initialized and moved to {device}")
# Load the saved model weights
try:
    model_loaded.load_state_dict(torch.load('face_expression_model.pth', map_location=device))
    model_loaded.eval()
    print(f"Model loaded successfully on {device}")
except FileNotFoundError:
    print("Error: Model file 'face_expression_model.pth' not found")
    raise
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load class labels
try:
    with open("Emotions.pkl", 'rb') as f:
        classes:list = pickle.load(f)
    print(f"Loaded {len(classes)} classes successfully")
except FileNotFoundError:
    print("Error: Classes file 'classes.pkl' not found")
    raise
except Exception as e:
    print(f"Error loading classes: {e}")
    raise

def predict(img_input):

    # Load and preprocess the image
    if isinstance(img_input, str):
        # Handle file path
        img = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, np.ndarray):
        # Handle numpy array (OpenCV image)
        if len(img_input.shape) == 3 and img_input.shape[2] == 3:
            # BGR to RGB conversion for OpenCV images
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_input.astype('uint8'), 'RGB')
    else:
        # Assume it's already a PIL Image
        img = img_input
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Load model weights
    # model.load_state_dict(torch.load('face_expression_model.pth'))
    # model.eval()
    
    with torch.no_grad():
        output :torch.Tensor = model_loaded(img_tensor)
        pred = output.argmax(dim=1)
    emotion = classes[pred.item()].split("__")[1]  # Extract emotion name from class label
    return {"emotion": emotion}



if __name__ == "__main__":
    # Example usage
    img_path = "data/test/7__Surprise/ffhq_400.png"  # Replace with your image path
    result = predict(img_path)
    print(f"Predicted emotion: {result['emotion']}")
    # Save the model and classes if needed