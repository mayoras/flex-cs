import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron consisting in two Fully-Connected hidden layers
    with non-linear activation function ReLU.
    """

    def __init__(self, in_features: int, hidden_layer: int, num_classes: int):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
