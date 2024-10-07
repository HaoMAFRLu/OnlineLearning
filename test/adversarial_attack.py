"""Test for different adversarail attack
"""
import os, sys
import torch
import random
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from online_attack import OnlineAttack
import utils as fcs

def test():
    root = fcs.get_parent_path(lvl=1)
    path = os.path.join(root, 'data', 
                        'newton_w_shift_wo_clear_wo_reset_padding',
                        '0.01_1.0_5.0', 'checkpoint_epoch_6000.pth')
    random.seed(9527)
    torch.manual_seed(9527)

    online_learning = OnlineAttack()
    online_learning.load_NN_model(path)
    online_learning.online_adversarial_attack(6000)

if __name__ == '__main__':
    test()