"""Classes for generating mnist data
"""
import numpy as np
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

random.seed(9527)

from mytypes import Array, Array2D

class MNISTGenerator():
    """
    """
    def __init__(self, device) -> None:
        self.load_data()
        self.distribution_initialization()

    def load_data(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

    def distribution_initialization(self):
        self.update_distribution([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    def update_distribution(self, dis: list, batch_size: int=1) -> None:
        distribution = np.array(dis)
        self.distribution = distribution / distribution.sum()
        self.batch_size = batch_size
        labels = self.get_label(self.train_dataset)
        weights = self.get_weights(labels, self.distribution)
        sampler = self.get_sampler(weights)
        self.dataloader = self.get_dataloader(self.batch_size, self.train_dataset, sampler)

    def get_sampler(self, weights):
        return torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=True
        )

    def get_dataloader(self, batch_size, dataset, sampler):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler
        )

    def get_weights(self, labels, distribution):
        label_counts = Counter(labels)
        class_sample_counts = np.array([label_counts[i] for i in range(10)])
        class_weights = distribution / class_sample_counts
        return class_weights[labels]

    def get_label(self, dataset):
        return dataset.targets.numpy()

        # test_dataset = datasets.MNIST(
        #     root='./data',
        #     train=False,
        #     download=True,
        #     transform=transform
        # )

        # train_loader = DataLoader(
        #     dataset=train_dataset,
        #     batch_size=64,
        #     shuffle=True
        # )

        # test_loader = DataLoader(
        #     dataset=test_dataset,
        #     batch_size=1000,
        #     shuffle=False
        # )

    def get_samples(self):
        data_iter = iter(self.dataloader)
        return next(data_iter)
        
