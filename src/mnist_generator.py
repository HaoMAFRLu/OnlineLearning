"""Classes for generating mnist data
"""
import numpy as np
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
import os, sys

random.seed(9527)

from mytypes import Array, Array2D
import utils as fcs
class MNISTGenerator():
    """
    """
    def __init__(self, mode) -> None:
        self.mode = mode
        root = fcs.get_parent_path(lvl=1)
        self.path = os.path.join(root, 'data')
        self.load_data()
        self.distribution_initialization()

    def load_data(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if self.mode == 'train':
            is_train = True
        elif self.mode == 'test':
            is_train = False

        self.dataset = datasets.MNIST(
                root=self.path,
                train=is_train,
                download=False,
                transform=transform
            )
        
    def distribution_initialization(self):
        self.update_distribution([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    def update_distribution(self, dis: list, batch_size: int=1) -> None:
        distribution = np.array(dis)
        self.distribution = distribution / distribution.sum()
        self.batch_size = batch_size
        labels = self.get_label(self.dataset)
        weights = self.get_weights(labels, self.distribution)
        sampler = self.get_sampler(weights)
        self.dataloader = self.get_dataloader(self.batch_size, self.dataset, sampler)

    def get_sampler(self, weights):
        return torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(weights),
            replacement=True
        )

    def get_dataloader(self, batch_size, dataset, sampler):
        if self.mode == 'train':
            _batch_size = batch_size
        elif self.mode == 'test':
            _batch_size = 10000        
        return DataLoader(
                dataset=dataset,
                batch_size=_batch_size,
                sampler=sampler
               )

    def get_weights(self, labels, distribution):
        label_counts = Counter(labels)
        class_sample_counts = np.array([label_counts[i] for i in range(10)])
        class_weights = distribution / class_sample_counts
        return class_weights[labels]

    def get_label(self, dataset):
        return dataset.targets.numpy()

    def get_samples(self):
        data_iter = iter(self.dataloader)
        return next(data_iter)
        
