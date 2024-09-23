import os, sys
from scipy.io import loadmat
import pickle 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

root = fcs.get_parent_path(lvl=1)
path_mat = os.path.join(root, 'src', 'data', 'linear_model', 'model_params.mat')
path_save = os.path.join(root, 'src', 'data', 'linear_model', 'linear_model')
data = loadmat(path_mat)

B = data['B']
Bd = data['Bd']

data = {
    'B': B,
    'Bd': Bd
}

with open(path_save, 'wb') as file:
    pickle.dump(data, file)