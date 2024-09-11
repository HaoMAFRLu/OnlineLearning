import numpy as np
import os, sys
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

root = fcs.get_parent_path(lvl=1)
file = os.path.join(root, 'params.txt')
alpha = ['0.1', '0.5', '1.0']
epsilon = ['0.1', '0.5', '1.0']
eta = ['0.5', '2.0', '5', '10']
value_ini = [f"{10**-i:.0e}" for i in range(3, 10)]

combinations = itertools.product(alpha, epsilon, eta)

# 打开文件以写入参数
with open(file, 'w') as f:
    for combo in combinations:
        f.write(" ".join(combo) + "\n")




