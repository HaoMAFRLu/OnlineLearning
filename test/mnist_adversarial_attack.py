"""Test for different adversarail attack
"""
import os, sys
import torch
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from mnist_online_attack import OnlineAttack
import utils as fcs

def test():
    root = fcs.get_parent_path(lvl=1)
    path = os.path.join(root, 'data', 
                        'mnist_w_shift2',
                        '0.1_0.1_0.05', 'checkpoint_epoch_50000.pth')
    random.seed(9527)
    torch.manual_seed(9527)

    online_learning = OnlineAttack(
        mode='newton',
        root_name='mnist_adversarial_attack',
        folder_name='test'
    )
    online_learning.load_NN_model(path)
    online_learning.online_adversarial_attack(10000, delta=3.0, 
                                              alpha=0.1, epsilon=0.1, eta=10.0)

if __name__ == '__main__':
    test()