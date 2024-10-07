"""Classes for online learning algorithm
"""
import torch
from pathlib import Path
import os, sys
import importlib
from typing import Tuple, List
import pickle
import numpy as np
import numba as nb
from datetime import datetime
import time
from scipy.signal import butter, filtfilt, freqz
import random

random.seed(10086)

import utils as fcs
from mytypes import Array, Array2D, Array3D

import networks
import data_process
import params
import environmnet
from trajectory import TRAJ
from attacker import Attacker

second_linear_output = []

class OnlineAttack():
    """Classes for online adversarial attack
    """
    def __init__(self, mode: str='gradient',
                 nr_interval: int=1000,
                 nr_data_interval: int=1,
                 root_name: str='test',
                 folder_name: str=None) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))
        self.root = fcs.get_parent_path(lvl=0)
        self.nr_interval = nr_interval
        self.nr_data_interval = nr_data_interval
        
        parent = fcs.get_parent_path(lvl=1)

        if folder_name is None:
            current_time = datetime.now()
            folder_name = current_time.strftime('%Y%m%d_%H%M%S')
        
        self.path_model = os.path.join(parent, 'data', root_name, folder_name)
        self.path_data = os.path.join(self.path_model, 'data')

        fcs.mkdir(self.path_model)
        fcs.mkdir(self.path_data)

        parent_dir = fcs.get_parent_path(lvl=1)
        fcs.copy_folder(os.path.join(parent_dir, 'src'), self.path_model)
        fcs.copy_folder(os.path.join(parent_dir, 'test'), self.path_model)
        
        self.initialization()

    @staticmethod
    def get_params(path: Path) -> Tuple[dict]:
        """Return the hyperparameters for each module

        parameters:
        -----------
        path: path to folder of the config file
        """
        PATH_CONFIG = os.path.join(path, 'config.json')
        PARAMS_LIST = ["SIM_PARAMS", 
                       "DATA_PARAMS", 
                       "NN_PARAMS"]
        params_generator = params.PARAMS_GENERATOR(PATH_CONFIG)
        params_generator.get_params(PARAMS_LIST)
        return (params_generator.PARAMS['SIM_PARAMS'],
                params_generator.PARAMS['DATA_PARAMS'],
                params_generator.PARAMS['NN_PARAMS'])

    def _env_initialization(self, model_name: str, PARAMS: dict):
        """
        """
        env = environmnet.BEAM(model_name, PARAMS)
        env.initialization()
        return env
    
    def env_initialization(self, PARAMS: dict) -> environmnet:
        """Initialize the simulation environment
        """
        model = 'BeamSystem_' + str(1)
        self.env = self._env_initialization(model, PARAMS)
        # self.envs[0] = self._env_initialization('control_system_medium', PARAMS)
        # self.envs[1] = self._env_initialization('control_system_large', PARAMS)

    def data_process_initialization(self, PARAMS: dict) -> None:
        """Initialize the data processor

        parameters:
        -----------
        PARAMS: hyperparameters
        """
        self.DATA_PROCESS = data_process.DataProcess('online', PARAMS)
    
    def NN_initialization(self, PARAMS: dict) -> None:
        """Build the model and load the pretrained weights
        """
        self.model = networks.NETWORK_CNN(self.device, PARAMS)
        self.model.build_network()
        
    def traj_initialization(self, distribution: str='original') -> None:
        """Create the class of reference trajectories
        """
        # self.traj = TRAJ(distribution)
        self.traj = TRAJ('v1')

    def load_dynamic_model(self) -> None:
        """Load the dynamic model of the underlying system,
        including the matrices B and Bd
        """
        path_file = os.path.join(self.root, 'data', 'linear_model', 'linear_model')
        with open(path_file, 'rb') as file:
            _data = pickle.load(file)
        
        self.B = _data['B']
        self.Bd = _data['Bd']
        # self.inv_B = np.linalg.inv(self.B)
        # self.pinv_B = np.linalg.pinv(self.B)
    
    def online_attacker_initialization(self, delta: float, length: int) -> None:
        """Initialize the kalman filter
        """
        self.online_attacker = Attacker(delta, length, self.B, alpha=0.1, epsilon=5.0)

    def initialization(self) -> torch.nn:
        """Initialize everything:
        (0. reload the module from another src path, and load the weights)
        1. generate parameters for each module
            |-- SIM_PARAMS: parameters for initializing the simulation
            |-- DATA_PARAMS: parameters for initializing the online data processor
            |-- NN_PARAMS: parameters for initializing the neural network
        2. load and initialize the simulation environment
        3. load and initialize the data process
        4. build and load the pretrained neural network
        """
        SIM_PARAMS, DATA_PARAMS, NN_PARAMS = self.get_params(self.root)
        self.load_dynamic_model()
        self.traj_initialization()
        self.env_initialization(SIM_PARAMS)
        self.data_process_initialization(DATA_PARAMS)
        self.NN_initialization(NN_PARAMS)
    
    def load_NN_model(self, path: Path, NN: torch.nn=None) -> None:
        """Load the model via specified path
        """
        checkpoint = torch.load(path)
        if NN is None:
            self.model.NN.load_state_dict(checkpoint['model_state_dict'])
        else:
            NN.load_state_dict(checkpoint['model_state_dict'])

    @staticmethod
    def get_loss(y1: np.ndarray, y2: np.ndarray) -> float:
        """Calculate the loss
        """
        return 0.5*np.linalg.norm(y1-y2)/len(y1)
        
    @staticmethod
    def tensor2np(a: torch.tensor) -> Array:
        """Covnert tensor to numpy
        """
        return a.squeeze().to('cpu').detach().numpy()
    
    def np2tensor(self, a: Array) -> torch.tensor:
        """Covnert numpy to tensor
        """        
        a_tensor = torch.from_numpy(a).to(self.device)
        return a_tensor
       
    def save_checkpoint(self, idx: int) -> None:
        """Save the model
        """
        checkpoint = {
            'epoch': idx,
            'model_state_dict': self.model.NN.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict()
        }
        path_checkpoint = os.path.join(self.path_model, f'checkpoint_epoch_{idx}.pth')
        torch.save(checkpoint, path_checkpoint)

    def save_data(self, idx: int, **kwargs) -> None:
        """Save the data
        """
        path_data = os.path.join(self.path_data, str(idx))
        with open(path_data, 'wb') as file:
            pickle.dump(kwargs, file)


    def get_matrix(self, l: int, i: int,
                   hl: int=100, hr: int=100) -> Array2D:
        Z = self.Z.clone()
        Z[:, i:i+hr+hl+1] = self.I
        return Z[:, hl:hl+l]
    
    def get_u(self, yref: Array, is_gradient: bool=False) -> Array:
        """
        """
        l = yref.shape[1] - 1

        self.model.NN.eval()
        self.model.NN.zero_grad()

        y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])
        y_tensor = torch.cat(y_processed, dim=0)

        if is_gradient is True:
            y_tensor.requires_grad = True
            u_tensor = self.model.NN(y_tensor.float())
            
            gradients = torch.autograd.grad(outputs=u_tensor, 
                                            inputs=y_tensor, 
                                            grad_outputs=torch.ones_like(u_tensor))

            a = gradients[0].squeeze().float()
            for i in range(l):
                self.par_pi_par_y[i, :] = torch.matmul(a[i, :].reshape(1, -1).float(), self.get_matrix(l, i))

        else:
            u_tensor = self.model.NN(y_tensor.float())
        
        u = self.tensor2np(u_tensor)
        return self.DATA_PROCESS.inverse_output(u)

    def _run_sim(self, env: environmnet, 
                 yref: Array2D, is_gradient: bool=False) -> Tuple:
        """
        """
        u = self.get_u(yref, is_gradient)
        yout, _ = env.one_step(u.flatten())
        loss = self.get_loss(yout.flatten(), yref[0, 1:].flatten())
        return yout, u, loss

    def online_adversarial_attack(self, nr_iterations: int=100) -> None:
        """
        """        
        self._online_attack(nr_iterations)

    def shift_distribution(self, distribution: str='v1'):
        """change the distribution
        """
        self.traj_initialization(distribution)
        yref_marker, _ = self.traj.get_traj()
        return yref_marker

    def yref_update(self, yref: Array2D, dy: Array2D) -> None:
        """Update the parameters of the neural network
        """
        return yref + dy
    
    def get_energy(self, d) -> float:
        return np.linalg.norm(d)
    
    def matrix_initialization(self, l: int, hl: int=100, hr: int=100) -> None:
        Z = np.zeros((hl+hl+1, l+hl+hr))
        I = np.eye(hl+hr+1)
        par_pi_par_y = np.zeros((l, l))
        self.par_pi_par_y = torch.from_numpy(par_pi_par_y).to(self.device).float()
        self.Z = torch.from_numpy(Z).to(self.device).float()
        self.I = torch.from_numpy(I).to(self.device).float()

    def _online_attack(self, nr_iterations: int=100, delta: float=1.0):
        """Pipeline for online adversarial attack using gradient information
        """
        yref_ini, _ = self.traj.get_traj()  # get a fixed reference trajectory
        yout_ini, _, _ = self._run_sim(self.env, yref_ini, is_gradient=False)
        
        self.online_attacker_initialization(delta, yref_ini.shape[1])
        self.matrix_initialization(yref_ini.shape[1]-1)

        for i in range(nr_iterations):
            tt = time.time()

            yref = self.yref_update(yref_ini, self.online_attacker.y)
            input_energy = self.get_energy(self.online_attacker.y.flatten())

            t1 = time.time()
            yout, u, _ = self._run_sim(self.env, yref, is_gradient=True)
            output_energy = self.get_energy((yout - yout_ini).flatten())
            tsim = time.time() - t1
            
            self.online_attacker.import_par_pi_par_y(self.par_pi_par_y)
            self.online_attacker.attack(yref[0, 1:], yout)

            ttotal = time.time() - tt
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                InputEnergy=[input_energy],
                OutputEnergy=[output_energy],
                Ttotal=[ttotal],
                Tsim=[tsim])
            
            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i,
                               u=u,
                               yref=yref,
                               yout=yout,
                               dy=self.online_attacker.y)
                
            # if (i+1) % self.nr_interval == 0:
            #     self.save_checkpoint(i+1)


            

    


        