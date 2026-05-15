import torch
import torch.nn as nn
import torch.optim as optim

from src.models.baseline_cnn import BaselineCNN
from src.utils.dataloader import get_dataloaders

# -----------------------------------
# Device Configuration
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

# -----------------------------------
# Load Dataset
# -----------------------------------

train_loader, test_loader = get_dataloaders()

# -----------------------------------
# Initialize Model
# -----------------------------------

model = BaselineCNN().to(device)

# -----------------------------------
# Loss Function
# -----------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------
# Optimizer
# -----------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------
# Training Config
# -----------------------------------

EPOCHS = 10

# -----------------------------------
# Training Loop
# -----------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for raw_images, _, labels in train_loader:

        raw_images = raw_images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(raw_images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {avg_loss:.4f}"
    )

# -----------------------------------
# Save Model
# -----------------------------------

torch.save(
    model.state_dict(),
    "baseline_model.pth"
)

print("Baseline Model Saved.")
print("Training Complete.")