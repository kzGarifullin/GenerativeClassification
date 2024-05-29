import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader


class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, num_classes)
        )
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


class Net(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim=128):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


def split_dataset(dataset, num_train_per_class, num_test_per_class=None):
    train_indices = []
    test_indices = []
    for i in range(10):
        indices = torch.where(dataset.targets == i)[0]
        train_indices.extend(indices[:num_train_per_class])
        if num_test_per_class:
            test_indices.extend(indices[num_train_per_class:num_train_per_class+num_test_per_class])
        else:
            test_indices.extend(indices[num_train_per_class:])
    return train_indices, test_indices
