import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# Define the model architecture (matching the one in your notebook)
class nn_model0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_units, kernel_size=2),  # Grayscale images
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=2),
            nn.ReLU()
        )
        
        # Calculating the flattened size
        dummy_input = torch.randn(1, 1, 28, 28)
        dummy_output = self.conv_layers(dummy_input)
        flattened_size = dummy_output.view(-1).shape[0]
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_size, out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn_model0(input_shape=1, hidden_units=10, output_shape=10)  # Adjust output_shape as per your number of classes
model.load_state_dict(torch.load('model_300.pth', map_location=device))
model.to(device)
model.eval()

# Image preprocessing function (including the scaling/transform as used in your notebook)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image as was done during training (if applicable)
    ])
    
    # Open the image
    img = Image.open(image_path)
    
    # Apply the transform to the image and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Invert the image (make the background black and object white)
    img_tensor = 1 - img_tensor  # Invert the color values (0 becomes 1 and 1 becomes 0)

    return img_tensor, img  # Return the original image for display

# Class labels
class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Function to predict the class of a given image and return the corresponding label
def predict(image_path):
    img_tensor, img = preprocess_image(image_path)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return class_labels[prediction], img_tensor, img

# Function to predict all images in the "bilder" folder and show what the AI sees
def predict_and_visualize_all_images(folder_path):
    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".webp"):  # Check for image files
            file_path = os.path.join(folder_path, filename)
            result, img_tensor, original_img = predict(file_path)
            
            # Display the original image for comparison
            plt.subplot(1, 2, 1)
            plt.imshow(original_img, cmap='gray')
            plt.title(f'Original Image: {filename}')
            plt.axis('off')
            
            # Display the inverted image
            img_np = img_tensor.squeeze(0).cpu().numpy()  # Convert tensor to numpy array for display
            img_np = img_np[0]  # Only take the first channel (grayscale)
            plt.subplot(1, 2, 2)
            plt.imshow(img_np, cmap='gray')
            plt.title(f'Predicted class: {result}')
            plt.axis('off')
            
            plt.show()

# Example usage for all images in "bilder" folder
predict_and_visualize_all_images('bilder')
