import torch
import torch.nn as nn
import torch.optim as optim

from src.models.hybrid_cnn import HybridCNN
from src.utils.dataloader import get_dataloaders
from src.training.trainer import train_one_epoch

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

model = HybridCNN().to(device)

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

    loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    alpha_value = model.fusion.alpha.item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {loss:.4f} | "
        f"Fusion Alpha: {alpha_value:.4f}"
    )

# -----------------------------------
# Save Model
# -----------------------------------

torch.save(
    model.state_dict(),
    "hybrid_model_2.pth"
)

print("Hybrid Model Saved.")
print("Training Complete.")