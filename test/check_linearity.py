"""Test for checking gradient
"""
import os, sys
import torch
import random
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from online_learning import OnlineLearning
import utils as fcs

import networks
import data_process
import params
import environmnet
from trajectory import TRAJ
from online_optimizer import OnlineOptimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = fcs.get_parent_path(lvl=1)

def build_model(PARAMS):
    """Build the model
    """
    model = networks.NETWORK_CNN(device, PARAMS)
    model.build_network()
    return model

def load_the_model(PARAMS, path):
    """Load the pretrained model
    """
    model = build_model(PARAMS)
    checkpoint = torch.load(path)
    model.NN.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_params():
    """Return the hyperparameters for each module

    parameters:
    -----------
    path: path to folder of the config file
    """
    PATH_CONFIG = os.path.join(root, 'src', 'config.json')
    PARAMS_LIST = ["DATA_PARAMS", 
                    "NN_PARAMS"]
    params_generator = params.PARAMS_GENERATOR(PATH_CONFIG)
    params_generator.get_params(PARAMS_LIST)
    return (params_generator.PARAMS['DATA_PARAMS'],
            params_generator.PARAMS['NN_PARAMS'])


def build_environment(PARAMS: dict) -> environmnet:
    """Initialize the simulation environment
    """
    return None
    # env = environmnet.BEAM('Control_System', PARAMS)
    # env.initialization()
    # return env

def traj_initialization(distribution: str='original'):
    """Create the class of reference trajectories
    """
    traj_generator = TRAJ(distribution)
    return traj_generator

def data_initialization(PARAMS):
    return data_process.DataProcess('online', PARAMS)

def initialization():
    DATA_PARAMS, NN_PARAMS = get_params()
    traj_generator = traj_initialization()
    data_processor = data_initialization(DATA_PARAMS)
    return NN_PARAMS, traj_generator, data_processor

def tensor2np(a: torch.tensor):
    """Covnert tensor to numpy
    """
    return a.squeeze().to('cpu').detach().numpy()

def _u_wo_gradient(data_processor, model, y):
    model.NN.eval()
    y_processed = data_processor.get_data(raw_inputs=y[0, 1:])
    y_tensor = torch.cat(y_processed, dim=0)
    u_tensor = model.NN(y_tensor.float())
    return tensor2np(u_tensor)

def get_matrix(i):
    Z = np.zeros((100+100+1, 550+100+100))
    I = np.eye(100+100+1)
    Z[:, i:i+201] = I
    return Z[:, 100:650]

def _u_w_gradient(data_processor, model, y):
    model.NN.eval()
    model.NN.zero_grad()

    y_processed = data_processor.get_data(raw_inputs=y[0, 1:])
    y_tensor = torch.cat(y_processed, dim=0)
    y_tensor.requires_grad = True
    
    u_tensor = model.NN(y_tensor.float())
    gradients = torch.autograd.grad(outputs=u_tensor, 
                                    inputs=y_tensor, 
                                    grad_outputs=torch.ones_like(u_tensor))

    a = gradients[0].squeeze().to('cpu').numpy()
    Jac = np.zeros((550, 550))
    for i in range(550):
        Jac[i, :] = a[i, :].reshape(1, -1)@get_matrix(i)

    return tensor2np(u_tensor), Jac

def get_u(data_processor, model, y, is_gradient):
    if is_gradient is True:
        u, Jac = _u_w_gradient(data_processor, model, y)
    elif is_gradient is False:
        u = _u_wo_gradient(data_processor, model, y)
        Jac = None
    return u, Jac

def get_plots(dus):
    num = len(dus)
    fig, axs = plt.subplots(num, 1, figsize=(20, 20))
    for i in range(num):
        du = dus[i]
        ax = axs[i]
        u = du[0]
        u_bar = du[1]
        fcs.set_axes_format(ax, r'Time index', r'Displacement')
        ax.plot(u, linewidth=1.0, linestyle='-', label='du')
        ax.plot(u_bar, linewidth=1.0, linestyle='-', label='du_bar')
        ax.legend(fontsize=14)
    plt.show()

def check_linearity(data_processor, model, yref, yref_noise):
    dy = (yref_noise-yref).flatten()
    u, Jac = get_u(data_processor, model, yref, is_gradient=True)
    u_noise, _ = get_u(data_processor, model, yref_noise, is_gradient=False)
    du = u_noise - u
    du_bar = Jac@dy[1:].reshape(-1, 1)
    l = np.linalg.norm(du_bar.flatten() - du.flatten())
    print(l)
    return (du.flatten(), du_bar.flatten())
        
def test():
    """main script
    1. load the neural network
    2. initialize the environmnet
    3. generate the trajectory
    4. add noise to the trajectory
    5. implement the trajectory
    """
    # folder1s = ['newton_wo_shift', 'BFS', 'DFS', 'DFS2']
    # folders = ['0.01_1.0_5.0', '0.01_1.0_5.0', '0.01_1.0_5.0', '0.01_5.0_5.0']
    folder1s = ['newton_wo_shift', 'BFS2', 'BFS2', 'BFS2', 'BFS2', 'BFS2']
    folders = ['0.01_1.0_5.0', '0.01_1.0_0.1', '0.01_1.0_0.5', '0.01_1.0_0.05', '0.01_1.0_1.0', '0.01_1.0_5.0']
    models = []
    l = []
    du = []
    
    NN_PARAMS, traj_generator, data_processor = initialization()
    for folder1, folder in zip(folder1s, folders):
        path = os.path.join(root, 'data', folder1, folder, 'checkpoint_epoch_4000.pth')
        models.append(load_the_model(NN_PARAMS, path))

    while 1:
        yref, _ = traj_generator.get_traj()
        yref_noise = fcs.add_noise(yref)
        du = []
        for i in range(len(models)):
            model = models[i]
            # l.append(check_linearity(data_processor, model, yref, yref_noise))
            du.append(check_linearity(data_processor, model, yref, yref_noise))
        print('------------------------------')
        get_plots(du)
    
if __name__ == '__main__':
    test()