"""Online learning with newton method on the cluster
"""
import os, sys
import torch
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from mnist_online_learning import MNISTOnlineLearning

def test():
    random.seed(9527)
    torch.manual_seed(9527)

    parser = argparse.ArgumentParser(description="Online MNIST Training")
    parser.add_argument('--alpha', type=float, required=True, help="Alpha")
    parser.add_argument('--epsilon', type=float, required=True, help="Epsilon")
    parser.add_argument('--eta', type=float, required=True, help="Eta")
    args = parser.parse_args()

    folder_name = str(args.alpha)+'_'+str(args.epsilon)+'_'+str(args.eta)

    mode = 'newton'
    is_shift_dis = False
    is_clear     = False
    is_reset     = False

    name1 = 'w_shift' if is_shift_dis is True else 'wo_shift'
    name2 = 'w_clear' if is_clear is True else 'wo_clear'
    name3 = 'w_reset' if is_reset is True else 'wo_reset'
    root_name = mode + '_' + name1 + '_' + name2 + '_' + name3

    online_learning = MNISTOnlineLearning(mode='newton',
                                     root_name='mnist_wo_shift',
                                     folder_name=folder_name,
                                     alpha=args.alpha,epsilon=args.epsilon,eta=args.eta)
    
    online_learning.online_learning(10000, 
                                    is_shift_dis=is_shift_dis, 
                                    is_clear=is_clear,
                                    is_reset=is_reset)

if __name__ == '__main__':
    test()