import torch
import matplotlib.pyplot as plt

from src.models.hybrid_cnn import HybridCNN
from src.utils.dataloader import get_dataloaders

classes = [
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu'
)

_, test_loader = get_dataloaders(
    batch_size=8
)

model = HybridCNN().to(device)

model.load_state_dict(
    torch.load('hybrid_model_2.pth')
)

model.eval()

raw_images, filtered_images, labels = next(
    iter(test_loader)
)

with torch.no_grad():

    outputs = model(
        raw_images.to(device),
        filtered_images.to(device)
    )

    _, preds = torch.max(outputs, 1)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12,6)
)

for i, ax in enumerate(axes.flat):

    img = raw_images[i].permute(
        1,
        2,
        0
    ).numpy()

    ax.imshow(img)

    ax.set_title(
        f'Pred: {classes[preds[i]]}\n'
        f'True: {classes[labels[i]]}'
    )

    ax.axis('off')

plt.tight_layout()

plt.show()