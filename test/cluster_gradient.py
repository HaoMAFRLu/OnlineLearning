"""Online learning with gradient descent on the cluster
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
    random.seed(9527)
    torch.manual_seed(9527)

    parser = argparse.ArgumentParser(description="Online Training")
    parser.add_argument('--eta', type=float, required=True, help="Eta")
    args = parser.parse_args()

    folder_name = str(args.eta)

    online_learning = OnlineLearning(mode='gradient',
                                     root_name='gradient_wo_shift',
                                     folder_name=folder_name,
                                     alpha=1.0,epsilon=1.0,eta=args.eta)
    
    online_learning.online_learning(6000, is_shift_dis=False)

if __name__ == '__main__':
    test()