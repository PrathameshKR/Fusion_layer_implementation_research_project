import torch

from src.models.hybrid_cnn import HybridCNN
from src.models.baseline_cnn import BaselineCNN

from src.utils.dataloader import get_dataloaders

# -----------------------------------
# Device
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------
# Load Data
# -----------------------------------

train_loader, test_loader = get_dataloaders()

# -----------------------------------
# Load Models
# -----------------------------------

hybrid_model = HybridCNN().to(device)

baseline_model = BaselineCNN().to(device)

# -----------------------------------
# Load Saved Weights
# -----------------------------------

hybrid_model.load_state_dict(
    torch.load("hybrid_model_2.pth")
)

baseline_model.load_state_dict(
    torch.load("baseline_model.pth")
)

# -----------------------------------
# Evaluation Function
# -----------------------------------

def evaluate_hybrid(model, loader):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for raw_images, filtered_images, labels in loader:

            raw_images = raw_images.to(device)

            filtered_images = filtered_images.to(device)

            labels = labels.to(device)

            outputs = model(
                raw_images,
                filtered_images
            )

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

# -----------------------------------
# Baseline Evaluation
# -----------------------------------

def evaluate_baseline(model, loader):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for raw_images, _, labels in loader:

            raw_images = raw_images.to(device)

            labels = labels.to(device)

            outputs = model(raw_images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

# -----------------------------------
# Run Evaluation
# -----------------------------------

hybrid_acc = evaluate_hybrid(
    hybrid_model,
    test_loader
)

baseline_acc = evaluate_baseline(
    baseline_model,
    test_loader
)

print(f"Hybrid Accuracy: {hybrid_acc:.2f}%")

print(f"Baseline Accuracy: {baseline_acc:.2f}%")