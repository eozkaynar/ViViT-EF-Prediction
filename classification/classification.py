import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
from module import ViViT
import matplotlib.pyplot as plt
# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 60
INPUT_SHAPE = (1, 28, 28, 28)  # (C, D, H, W)
PATCH_SIZE = (8, 8, 8)
NUM_CLASSES = 11
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

# Dataset preparation using MedMNIST
def prepare_medmnist_data():
    info = INFO["organmnist3d"]
    DataClass = getattr(medmnist, info["python_class"])
    train_dataset = DataClass(split="train", download=True)
    val_dataset = DataClass(split="val", download=True)
    test_dataset = DataClass(split="test", download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

# Training Loop
def train_model():
    train_loader, val_loader, test_loader = prepare_medmnist_data()

    model = ViViT(input_shape=INPUT_SHAPE, patch_size=PATCH_SIZE, embed_dim=PROJECTION_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch, (data, labels) in enumerate(train_loader):
            data = data.float()
            labels = labels[:, 0].long()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.float()
                labels = labels[:, 0].long()
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    # Test phase
    model.eval()
    total_test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float()
            labels = labels[:, 0].long()
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig('training_validation_loss_plot.png')

    plt.show()

    return model
