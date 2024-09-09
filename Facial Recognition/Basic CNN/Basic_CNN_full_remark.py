# Step 1: Import Libraries
import torch  # Core PyTorch library for tensor operations and autograd
import torchvision  # For loading datasets and transformations
from torchvision import datasets, transforms  # Import specific modules from torchvision
from torch.utils.data import DataLoader  # For creating data loaders
import torch.nn as nn  # For building neural network layers
import torch.nn.functional as F  # For activation functions like ReLU
from sklearn.preprocessing import LabelEncoder  # For converting labels to numeric format

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
print(f"Using device: {device}")  # Print which device (CPU or GPU) is being used

# Step 2: Define image transformations (resize and normalize)
transform = transforms.Compose([  # Compose several image transformations into one
    transforms.Resize((64, 64)),   # Resize images to 64x64 pixels for faster processing
    transforms.ToTensor(),         # Convert image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range for each color channel
])

# Step 3: Load the LFW dataset for people (target_type='identity' for classification)
# LFW (Labeled Faces in the Wild) dataset, used for facial recognition tasks
lfw_dataset = datasets.LFWPeople(root='./data', split='train', download=True, transform=transform)  # Download and apply transformation

# Step 4: Label Encoding (convert string labels to integers)
le = LabelEncoder()  # Create a label encoder to convert string labels to integers
lfw_dataset.targets = le.fit_transform(lfw_dataset.targets)  # Convert target labels (person names) to numeric format

# Step 5: Create DataLoader
data_loader = DataLoader(lfw_dataset, batch_size=64, shuffle=True)  # Create a DataLoader for batching and shuffling the data

# Step 6: Define the Expanded CNN Model
class ExpandedCNN(nn.Module):  # Define a neural network class, inheriting from nn.Module
    def __init__(self):
        super(ExpandedCNN, self).__init__()  # Initialize the base class
        # Convolutional layers (extract features from images)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # First convolutional layer, 32 filters, 3x3 kernel
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for the first conv layer

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Second conv layer, 64 filters
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for the second conv layer

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Third conv layer, 128 filters
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization for the third conv layer

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Fourth conv layer, 256 filters
        self.bn4 = nn.BatchNorm2d(256)  # Batch normalization for the fourth conv layer

        # Fully connected (FC) layers (for final classification)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # First FC layer, input size corresponds to the flattened output of the conv layers
        self.fc2 = nn.Linear(512, 128)  # Second FC layer, reducing size to 128
        self.fc3 = nn.Linear(128, len(set(lfw_dataset.targets)))  # Output layer, number of classes equals the number of unique identities

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)  # 50% of neurons will be randomly dropped during training

    def forward(self, x):
        # Define the forward pass (how the data flows through the layers)
        x = F.relu(self.bn1(self.conv1(x)))  # First conv layer followed by ReLU activation and batch normalization
        x = F.max_pool2d(x, 2)  # Max pooling layer to reduce spatial dimensions

        x = F.relu(self.bn2(self.conv2(x)))  # Second conv layer followed by ReLU and batch normalization
        x = F.max_pool2d(x, 2)  # Max pooling

        x = F.relu(self.bn3(self.conv3(x)))  # Third conv layer with ReLU and batch normalization
        x = F.max_pool2d(x, 2)  # Max pooling

        x = F.relu(self.bn4(self.conv4(x)))  # Fourth conv layer with ReLU and batch normalization
        x = F.max_pool2d(x, 2)  # Max pooling

        # Flatten the output tensor from the convolutional layers into a 1D vector
        x = x.view(-1, 256 * 4 * 4)  # Reshape to (batch_size, flattened size)

        # Fully connected layers
        x = F.relu(self.fc1(x))  # First FC layer followed by ReLU
        x = self.dropout(x)  # Apply dropout to prevent overfitting
        x = F.relu(self.fc2(x))  # Second FC layer with ReLU
        x = self.dropout(x)  # Apply dropout again
        x = self.fc3(x)  # Output layer, no activation since we're using cross-entropy loss

        return x  # Return the output

# Step 7: Initialize model, loss function, and optimizer
expanded_model = ExpandedCNN().to(device)  # Move the model to the GPU (if available)
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = torch.optim.Adam(expanded_model.parameters(), lr=0.001)  # Adam optimizer with learning rate of 0.001

# Step 8: Training Loop with GPU Support
num_epochs = 10  # You can adjust the number of epochs (iterations over the full dataset)

for epoch in range(num_epochs):
    running_loss = 0.0  # Track running loss for the epoch
    for images, labels in data_loader:  # Loop through batches of images and labels
        images, labels = images.to(device), labels.to(device)  # Move data to GPU (or CPU if unavailable)

        optimizer.zero_grad()  # Clear the gradients from the previous iteration

        # Forward pass
        outputs = expanded_model(images)  # Get model predictions

        # Compute loss
        loss = criterion(outputs, labels)  # Calculate the loss between predictions and actual labels

        # Backward pass and optimization
        loss.backward()  # Compute gradients with backpropagation
        optimizer.step()  # Update the model's parameters

        # Track running loss
        running_loss += loss.item()  # Add loss for this batch to the running total

    # Print loss at the end of each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')  # Average loss for the epoch

# Step 9: Evaluate Model on GPU
def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode (disables dropout and other training-specific behaviors)
    correct = 0  # Counter for correct predictions
    total = 0  # Total number of samples
    with torch.no_grad():  # Disable gradient calculation for evaluation (saves memory and computation)
        for images, labels in data_loader:  # Loop through batches of data
            images, labels = images.to(device), labels.to(device)  # Move data to GPU (or CPU)
            outputs = model(images)  # Get model predictions
            _, predicted = torch.max(outputs, 1)  # Get the index of the highest score for each sample (the predicted class)
            total += labels.size(0)  # Update total samples count
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Print accuracy as a percentage
    print(f'Accuracy: {100 * correct / total:.2f}%')

# Evaluate model accuracy on the training set
evaluate_model(expanded_model, data_loader)  # Call the evaluation function to test the model's performance
