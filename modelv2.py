import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from private_values import training_root, testing_root
import torchvision
import multiprocessing


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_classes = 260

weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
preprocess = weights.transforms()

model = torchvision.models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.to(device)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
) # learn lamba functions

if __name__ == "__main__":
    train_dir = training_root
    test_dir = testing_root

    multiprocessing.set_start_method('spawn', force=True)
    # enter the filepath to the data from fruits-360
    # downloading fruits datasets
    train_ds = datasets.ImageFolder(root=train_dir, transform=preprocess)
    test_ds = datasets.ImageFolder(root=test_dir, transform=preprocess)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4,persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True)

    criterion = nn.CrossEntropyLoss()
    print("Training Started...")


    for epoch in range(10):
        for images, labels in train_loader:
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            model_outputs = model(images)

            loss = criterion(model_outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            total_val_loss = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item() * images.size(0)

            average_val_loss = total_val_loss / len(test_loader.dataset)

        print(f"Epoch: {epoch + 1}, Loss: {loss:.4f}, Val_loss: {average_val_loss:.4f}")

    print("Training Complete!")

    torch.save(model, "fruit_modelV2.pt")
