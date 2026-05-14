import torch
import torch.nn as nn
import torch.optim as optim

from src.models.hybrid_cnn import HybridCNN

from src.utils.dataloader import get_dataloaders

from src.training.trainer import train_one_epoch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_loader, test_loader = get_dataloaders()

model = HybridCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 10

for epoch in range(EPOCHS):

    loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")