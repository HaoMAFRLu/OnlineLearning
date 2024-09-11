"""Test for online training
"""
import os, sys
import torch
import random
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from online_learning import OnlineLearning
import utils as fcs

def test():
    root = fcs.get_parent_path(lvl=0)
    path = os.path.join(root, 'config.json')

    random.seed(9527)
    torch.manual_seed(9527)

    online_learning = OnlineLearning(mode='newton', 
                                     alpha=0.1,epsilon=1.0,eta=10)
    online_learning.online_learning(6000, is_shift_dis=True)

if __name__ == '__main__':
    test()