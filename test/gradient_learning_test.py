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

def load_the_model(PARAMS):
    """Load the pretrained model
    """

    model = build_model(PARAMS)
    folder = 'newton_w_shift_wo_clear_wo_reset_padding'
    file = '0.01_1.0_5.0'
    path_model = os.path.join(root, 'data', folder, file, 'checkpoint_epoch_6000.pth')
    checkpoint = torch.load(path_model)
    model.NN.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_params():
    """Return the hyperparameters for each module

    parameters:
    -----------
    path: path to folder of the config file
    """
    PATH_CONFIG = os.path.join(root, 'src', 'config.json')
    PARAMS_LIST = ["SIM_PARAMS", 
                    "DATA_PARAMS", 
                    "NN_PARAMS"]
    params_generator = params.PARAMS_GENERATOR(PATH_CONFIG)
    params_generator.get_params(PARAMS_LIST)
    return (params_generator.PARAMS['SIM_PARAMS'],
            params_generator.PARAMS['DATA_PARAMS'],
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
    SIM_PARAMS, DATA_PARAMS, NN_PARAMS = get_params()
    model = load_the_model(NN_PARAMS)
    env = build_environment(SIM_PARAMS)
    traj_generator = traj_initialization()
    data_processor = data_initialization(DATA_PARAMS)
    return model, env, traj_generator, data_processor

def tensor2np(a: torch.tensor):
    """Covnert tensor to numpy
    """
    return a.squeeze().to('cpu').detach().numpy()

def get_u(data_processor, model, y):
    model.NN.eval()

    y_processed = data_processor.get_data(raw_inputs=y[0, 1:])
    y_tensor = torch.cat(y_processed, dim=0)
    u_tensor = model.NN(y_tensor.float())

    return tensor2np(u_tensor)

def run_sim(env, u):
    yout, _ = env.one_step(u.flatten())
    return yout

def get_plots(**kwargs):
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    ax = axs[0]
    fcs.set_axes_format(ax, r'Time index', r'Displacement')
    ax.plot(kwargs['yref'].flatten()[1:], linewidth=1.0, linestyle='-', label='yref')
    ax.plot(kwargs['yref_noise'].flatten()[1:], linewidth=1.0, linestyle='-', label='yref_noise')
    ax.plot(kwargs['yout_noise'].flatten()[0:], linewidth=1.0, linestyle='-', label='yout_noise')
    ax.legend(fontsize=14)

    ax = axs[1]
    fcs.set_axes_format(ax, r'Time index', r'Displacement')
    ax.plot(kwargs['u'].flatten()[1:], linewidth=1.0, linestyle='-', label='u')
    ax.plot(kwargs['u_noise'].flatten()[1:], linewidth=1.0, linestyle='-', label='u_noise')
    ax.legend(fontsize=14)
    plt.show()

def test():
    """main script
    1. load the neural network
    2. initialize the environmnet
    3. generate the trajectory
    4. add noise to the trajectory
    5. implement the trajectory
    """
    model, env, traj_generator, data_processor = initialization()
    yref, _ = traj_generator.get_traj()

    while 1:
        yref, _ = traj_generator.get_traj()
        yref_noise = fcs.add_noise(yref)
        u_noise = get_u(data_processor, model, yref_noise)
        u = get_u(data_processor, model, yref)
        # yout_noise = run_sim(env, u_noise)
        get_plots(yref=yref, yref_noise=yref_noise,
                yout_noise=yref_noise, u=u, u_noise=u_noise)

if __name__ == '__main__':
    test()