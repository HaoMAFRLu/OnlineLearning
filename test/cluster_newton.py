"""Online learning with newton method on the cluster
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
    parser.add_argument('--alpha', type=float, required=True, help="Alpha")
    parser.add_argument('--epsilon', type=float, required=True, help="Epsilon")
    parser.add_argument('--eta', type=float, required=True, help="Eta")
    args = parser.parse_args()

    folder_name = str(args.alpha)+'_'+str(args.epsilon)+'_'+str(args.eta)

    online_learning = OnlineLearning(mode='newton',
                                     root_name='newton_w_shift_wo_clear_wo_reset',
                                     folder_name=folder_name,
                                     alpha=args.alpha,epsilon=args.epsilon,eta=args.eta)
    
    online_learning.online_learning(6000, 
                                    is_shift_dis=True, 
                                    is_clear=False,
                                    is_reset=False)

if __name__ == '__main__':
    test()