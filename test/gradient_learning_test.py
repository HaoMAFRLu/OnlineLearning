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
    env = environmnet.BEAM('Control_System', PARAMS)
    env.initialization()
    return env

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

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def add_noise(y, snr_db=1):
    # noise = np.random.normal(0, 0.1, size=y.shape)
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, size=y.shape)
    noise[0, 0] = 0.0
    noise[0, -51:] = 0.0

    cutoff = 10 # 截止频率（Hz）
    order = 4
    fs = 500
    b, a = butter_lowpass(cutoff, fs, order)
    filtered_noise = filtfilt(b, a, noise)
    
    return y + filtered_noise

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
        yref_noise = add_noise(yref)
        u_noise = get_u(data_processor, model, yref_noise)
        u = get_u(data_processor, model, yref)
        yout_noise = run_sim(env, u_noise)
        get_plots(yref=yref, yref_noise=yref_noise,
                yout_noise=yout_noise, u=u, u_noise=u_noise)
        
        yref = yref_noise

if __name__ == '__main__':
    test()