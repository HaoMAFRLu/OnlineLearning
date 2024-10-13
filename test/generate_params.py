import numpy as np
import os, sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

root = fcs.get_parent_path(lvl=1)
file = os.path.join(root, 'params_mnist.txt')
alpha = ['0.01', '0.05', '0.1', '0.5', '1.0', '5.0']
epsilon = ['0.01', '0.05', '0.1', '0.5', '1.0', '5.0']
multipliers = ['0.5', '1.0', '5.0', '10.0']

combinations = [(a, e, str(float(e)*float(i))) for a in alpha for e in epsilon for i in multipliers]

with open(file, 'w') as f:
    for combo in combinations:
        f.write(" ".join(combo) + "\n")




