import math
from typing import Hashable

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split

from flex.data import Dataset, FedDatasetConfig, FedDataDistribution

mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

cifar_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def split_train_val(train_data: datasets.vision.VisionDataset, val_size: float):
    val_len = math.floor(val_size * len(train_data))
    train_len = len(train_data) - val_len

    train_data, val_data = random_split(train_data, [train_len, val_len])

    return train_data, val_data


def federated_torch_dataset(
    dataset: datasets.vision.VisionDataset,
    n_nodes: int,
    seed: int = 33,
    server_id: Hashable = "server",
    return_val_data: bool = False,
    val_size: float = 0.2,
):
    train_data = dataset(
        root="../data",
        train=True,
        download=True,
        transform=None,  # we apply them later in training process
    )

    test_data = dataset(
        root="../data",
        train=False,
        download=True,
        transform=None,  # we apply them later in training process
    )

    config = FedDatasetConfig(seed=seed)
    config.replacement = False
    config.n_nodes = n_nodes

    # split train data into training and validation datasets
    if return_val_data:
        train_data, val_data = split_train_val(train_data, val_size)

    flex_dataset = FedDataDistribution.from_config(
        centralized_data=Dataset.from_torchvision_dataset(train_data), config=config
    )

    # assign test data to server_id
    server_id = "server"
    flex_dataset[server_id] = Dataset.from_torchvision_dataset(test_data)

    return flex_dataset, val_data if return_val_data else flex_dataset
