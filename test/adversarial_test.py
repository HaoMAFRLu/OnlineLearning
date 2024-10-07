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

def env_initialization(model_name: str, PARAMS: dict):
        """
        """
        env = environmnet.BEAM(model_name, PARAMS)
        env.initialization()
        return env
    
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
    env = environmnet.BEAM('BeamSystem_1', PARAMS)
    env.initialization()
    return env

def traj_initialization(distribution: str='v1'):
    """Create the class of reference trajectories
    """
    traj_generator = TRAJ(distribution)
    return traj_generator

def data_initialization(PARAMS):
    return data_process.DataProcess('online', PARAMS)

def initialization():
    SIM_PARAMS,DATA_PARAMS, NN_PARAMS = get_params()
    env = build_environment(SIM_PARAMS)
    traj_generator = traj_initialization()
    data_processor = data_initialization(DATA_PARAMS)
    return NN_PARAMS, env, traj_generator, data_processor

def tensor2np(a: torch.tensor):
    """Covnert tensor to numpy
    """
    return a.squeeze().to('cpu').detach().numpy()

def run_sim(env: environmnet, u):
    """
    """
    yout, _ = env.one_step(u.flatten())
    return yout

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

def get_u(data_processor, model, y, is_gradient=False):
    if is_gradient is True:
        u, Jac = _u_w_gradient(data_processor, model, y)
    elif is_gradient is False:
        u = _u_wo_gradient(data_processor, model, y)
        Jac = None
    return u, Jac

def get_plots(**kwargs):
    num = len(kwargs)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    fcs.set_axes_format(ax, r'Time index', r'Displacement')
    for key, value in kwargs.items():
        ax.plot(value, linewidth=1.0, linestyle='-', label=key)
    ax.legend(fontsize=14)
    plt.show()

def get_loss(y1, y2):
    return np.linalg.norm(y1-y2)

def retain_top_n(arr, n):
    """
    保留数组 arr 中绝对值最大的 n 个元素，其余元素设为 0。

    参数：
    - arr: 输入的 NumPy 数组
    - n: 要保留的元素个数

    返回：
    - result: 处理后的 NumPy 数组
    """
    # 检查 n 是否有效
    if n <= 0:
        raise ValueError("n 必须是正整数")
    if n > arr.size:
        n = arr.size

    # 计算绝对值
    abs_arr = np.abs(arr)
    
    # 找到绝对值最大的 n 个元素的索引
    indices = np.argpartition(-abs_arr.flatten(), n - 1)[:n]
    
    # 创建结果数组
    result = np.zeros_like(arr).flatten()
    result[indices] = arr.flatten()[indices]
    result = result.reshape(arr.shape)
    
    return result

def test():
    """main script
    1. load the neural network
    2. initialize the environmnet
    3. generate the trajectory
    4. adversarial attack
    """
    folder1 = 'newton_w_shift_wo_clear_wo_reset_padding'
    folder = '0.01_1.0_5.0'
    
    NN_PARAMS, env, traj_generator, data_processor = initialization()
    path = os.path.join(root, 'data', folder1, folder, 'checkpoint_epoch_6000.pth')
    model = load_the_model(NN_PARAMS, path)
    yref, _ = traj_generator.get_traj()
    yref_ini = yref.copy()

    for _ in range(20):
        u, _ = get_u(data_processor, model, yref)
        yout = run_sim(env, u)
        # get_plots(yref=yref[0, 1:].flatten(), 
        #           yref_ini=yref_ini[0, 1:].flatten(), 
        #           yout=yout.flatten())
        loss = get_loss(yref[0, 1:].flatten(), yout.flatten())
        print(loss)
        yref = np.insert(yout.flatten(), 0, 0.0, axis=None).reshape(1, -1)
    
    dy = yref - yref_ini
    dy_filtered = retain_top_n(dy, round(dy.size/5))
    yref = yref_ini + dy_filtered
    u, _ = get_u(data_processor, model, yref)
    yout = run_sim(env, u)
    get_plots(yref=yref[0, 1:].flatten(), 
              yout=yout.flatten())
    loss = get_loss(yref[0, 1:].flatten(), yout.flatten())
    print(np.linalg.norm(dy_filtered))
    print(loss)
    

    
if __name__ == '__main__':
    test()