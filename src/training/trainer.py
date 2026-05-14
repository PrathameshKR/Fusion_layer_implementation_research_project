import torch

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0

    for raw_images, filtered_images, labels in train_loader:

        raw_images = raw_images.to(device)

        filtered_images = filtered_images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            raw_images,
            filtered_images
        )

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)