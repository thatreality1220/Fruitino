import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from private_values import training_root, testing_root

train_dir = training_root
test_dir = testing_root 
# enter the filepath to the data from fruits-360

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# downloading fruits datasets
train_ds = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_ds = datasets.ImageFolder(root=test_dir, transform=test_transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

device = torch.device("mps")

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),

    nn.Flatten(),
    nn.Linear(9216, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 131)
).to(device)

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

print("Training Starting...")
for epoch in range(15):
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        model_outputs = model(images)

        loss = criterion(model_outputs, labels)

        loss.backward()
        optimizer.step()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item() * images.size(0)

        average_val_loss = total_val_loss / len(train_loader.dataset)

    print(f"Epoch: {epoch + 1}, Loss: {loss}, Val_loss: {average_val_loss}")

print("Training Complete!")

torch.save(model, "fruit_model.pt")
