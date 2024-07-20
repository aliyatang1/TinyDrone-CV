import os
import torch
from torchvision import transforms
from PIL import Image
from monodepth2 import networks
from monodepth2.layers import DispToDepth

def load_model(model_path):
    model = networks.ResnetEncoder(18, False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(192),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(input_image).unsqueeze(0)

def estimate_depth(model, image_tensor):
    with torch.no_grad():
        features = model.encode(image_tensor)
        depth = model.decode(features)
        depth = DispToDepth()(depth)
        return depth

if __name__ == "__main__":
    model_path = 'models/monodepth2_model.pth'
    image_path = 'path_to_your_image.jpg'
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    depth = estimate_depth(model, image_tensor)

    # Process and visualize depth map
    # You might need to convert tensor to an image and save it or display it using matplotlib
