from functools import reduce

from flex.pool import FlexPool
from .model import MLP
from .dataset import federated_torch_dataset
from .server import build_server_model
from .algorithms import random_sampling_run

import torchvision.datasets as datasets

if __name__ == "__main__":
    # Multi-Layer Perceptron
    def mnist_model():
        mnist_dims = (28, 28)
        mnist_classes = 10

        in_features = reduce(lambda x, y: x * y, mnist_dims)
        hidden_layer = 128

        return MLP(in_features, hidden_layer, mnist_classes)

    # Create a dataset
    n_nodes = 1000
    flex_dataset = federated_torch_dataset(datasets.MNIST, n_nodes=1000)

    # Create a pool
    flex_pool = FlexPool.client_server_pool(
        flex_dataset, init_func=build_server_model(mnist_model)
    )

    # Train
    losses, accuracies = random_sampling_run(flex_pool, rounds=20, clients_per_round=20)
