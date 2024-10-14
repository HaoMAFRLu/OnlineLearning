"""Test for online training
"""
import os, sys
import torch
import random
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from mnist_online_learning import MNISTOnlineLearning
import utils as fcs

def test():
    random.seed(9527)
    torch.manual_seed(9527)

    online_learning = MNISTOnlineLearning(mode='newton',
                                         root_name='test', 
                                         alpha=0.01,epsilon=1.0,eta=0.5,gamma=0.1)
    
    online_learning.online_learning(6000, 
                                    is_shift_dis=False,
                                    is_clear=False,
                                    is_reset=False)

if __name__ == '__main__':
    test()