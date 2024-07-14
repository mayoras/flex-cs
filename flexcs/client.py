from flex.model import FlexModel
from flex.data import Dataset

from torch.utils.data import DataLoader
from torchvision import transforms

from .common import Device


def train(
    client_flex_model: FlexModel,
    client_data: Dataset,
    epochs: int,
    batch_size: int,
    transforms: transforms.Compose,
    device: Device,
):
    train_dataset = client_data.to_torchvision_dataset(transform=transforms)
    client_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    for _ in range(epochs):
        for imgs, labels in client_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
