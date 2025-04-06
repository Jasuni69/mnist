import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Detailed device detection
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("ROCm/HIP available:", hasattr(torch.version, 'hip') and torch.version.hip is not None)

# Set device
if hasattr(torch.version, 'hip') and torch.version.hip is not None:
    device = torch.device("cuda")  # ROCm uses "cuda" as device name
    print("Using ROCm GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"Device being used: {device}")

# Define the neural network
class MNISTPerceptron(nn.Module):
    def __init__(self):
        super(MNISTPerceptron, self).__init__()
        # Input layer: 28x28 = 784 pixels
        # Hidden layer: 128 neurons
        # Output layer: 10 classes (digits 0-9)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 512)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        return x

def load_data():
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move data to GPU
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                avg_loss = running_loss / 100
                losses.append(avg_loss)
                print(f'[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}')
                running_loss = 0.0
    return losses

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to GPU
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

def plot_training(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Batch (x100)')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()

def main():
    # Initialize the model, loss function, and optimizer
    model = MNISTPerceptron().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    losses = train(model, train_loader, criterion, optimizer, epochs=5)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Test the model
    print("\nTesting model...")
    accuracy = test(model, test_loader)
    
    # Plot training loss
    plot_training(losses)
    
    print(f"\nFinal Results:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 