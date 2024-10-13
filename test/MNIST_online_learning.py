"""Test script for online learning for MNIST dataset
"""
import os, sys
from collections import Counter
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from mnist_generator import MNISTGenerator


def test():
    nr_samples = 1000
    batch_size = 1
    distribution = [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    data_generator = MNISTGenerator()
    data_generator.update_distribution(distribution, batch_size)
    
    labels = np.array([])
    for _ in range(nr_samples):
        _, _labels = data_generator.get_samples()
        labels = np.concatenate((labels, _labels.numpy()))
    
    batch_label_counts = Counter(labels)
    batch_label_counts_array = np.array([batch_label_counts.get(i, 0) for i in range(10)])
    batch_label_distribution = batch_label_counts_array / (batch_size*nr_samples)
    print("批次中各类别样本数量：", batch_label_counts)
    print("批次中各类别样本比例：", batch_label_distribution)
    print("目标标签分布：", distribution)

if __name__ == '__main__':
    test()