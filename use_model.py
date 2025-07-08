import pickle
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load the model architecture
model_loaded =  nn.Sequential(
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
        classes = pickle.load(f)
    print(f"Loaded {len(classes)} classes successfully")
except FileNotFoundError:
    print("Error: Classes file 'classes.pkl' not found")
    raise
except Exception as e:
    print(f"Error loading classes: {e}")
    raise

def predict(img_path):
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Load model weights
    # model.load_state_dict(torch.load('face_expression_model.pth'))
    # model.eval()
    
    with torch.no_grad():
        output = model_loaded(img_tensor)
        pred = output.argmax(dim=1)
    emotion = classes[pred.item()].split("__")[1]  # Extract emotion name from class label
    return {"emotion": emotion}



if __name__ == "__main__":
    # Example usage
    img_path = "data/test/7__Surprise/ffhq_400.png"  # Replace with your image path
    result = predict(img_path)
    print(f"Predicted emotion: {result['emotion']}")
    # Save the model and classes if needed