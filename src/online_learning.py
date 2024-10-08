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
from online_optimizer import OnlineOptimizer

second_linear_output = []

class OnlineLearning():
    """Classes for online learning
    """
    def __init__(self, mode: str='gradient',
                 nr_interval: int=1000,
                 nr_shift_dis: int=3000,
                 nr_data_interval: int=1,
                 nr_marker_interval: int=20,
                 root_name: str='test',
                 folder_name: str=None,
                 alpha: float=None,
                 epsilon: float=None,
                 eta: float=None,
                 gamma: float=None) -> None:
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(0))
        self.root = fcs.get_parent_path(lvl=0)
        self.nr_interval = nr_interval
        self.nr_shift_dis = nr_shift_dis
        self.nr_data_interval = nr_data_interval
        self.nr_marker_interval = nr_marker_interval
        self.mode = mode
        self.alpha = alpha
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma

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
        # nr_models = 1
        # self.envs = [None] * nr_models
        # for i in range(nr_models):
        #     model = 'BeamSystem_' + str(i+1)
        #     self.envs[i] = self._env_initialization(model, PARAMS)

        self.envs = [None] * 2
        self.envs[0] = self._env_initialization('control_system_medium', PARAMS)
        self.envs[1] = self._env_initialization('control_system_large', PARAMS)

    def data_process_initialization(self, PARAMS: dict) -> None:
        """Initialize the data processor

        parameters:
        -----------
        PARAMS: hyperparameters
        """
        self.DATA_PROCESS = data_process.DataProcess('online', PARAMS)
    
    def NN_initialization(self, path: Path, PARAMS: dict) -> None:
        """Build the model and load the pretrained weights
        """
        self.model = networks.NETWORK_CNN(self.device, PARAMS)
        self.model.build_network()
        
        if path is not None:
            checkpoint = torch.load(path)
            self.model.NN.load_state_dict(checkpoint['model_state_dict'])

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
    
    def online_optimizer_initialization(self) -> None:
        """Initialize the kalman filter
        """
        self.online_optimizer = OnlineOptimizer(mode=self.mode, B=self.B, 
                                                alpha=self.alpha, epsilon=self.epsilon,
                                                eta=self.eta, gamma=self.gamma)

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
        self.NN_initialization(None, NN_PARAMS)
        self.online_optimizer_initialization()
    
    # @nb.jit(nopython=True)
    def get_u(self, y: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Get the input u based on the disturbance
        """
        return self.inv_B@(y-self.Bd@d)

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
    
    def extract_last_layer(self, NN: torch.nn) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract the last layer of the neural network
        """
        last_layer = NN.fc[-1]
        return last_layer.weight.data, last_layer.bias.data

    def extract_last_layer_vec(self, NN: torch.nn) -> torch.Tensor:
        """Extract the last layer and vectorize them
        """
        w, b = self.extract_last_layer(NN)
        return torch.cat((w.t().flatten(), b.flatten()), dim=0).view(-1, 1)
    
    def extract_output(self) -> torch.Tensor:
        """
        """
        return second_linear_output[-1]

    def extract_NN_info(self, NN: torch.nn) -> Tuple[Array, Array]:
        """Extract the infomation of the neural network

        parameters:
        -----------
        NN: the given neural network

        returns:
        --------
        phi: the output of the second last layer
        vec: the column vector of the parameters of the last layer,
           including the bias
        """
        vec = self.extract_last_layer_vec(NN)
        phi = self.extract_output()        
        return phi, vec

    def _recover_last_layer(self, value: torch.Tensor, num: int) -> None:
        """
        """
        w = value[0:num].view(-1, 550).t()
        b = value[num:].flatten()
        return w, b        

    def assign_last_layer(self, NN: torch.nn, value: torch.Tensor) -> None:
        """Assign the value of the last layer of the neural network.
        """
        last_layer = NN.fc[-1]
        num = last_layer.weight.numel()
        w, b = self._recover_last_layer(value, num)
        
        with torch.no_grad():
            last_layer.weight.copy_(w)
            last_layer.bias.copy_(b)

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
    
    def get_par_pi_par_omega(self, NN:torch.nn, outputs: torch.Tensor):
        jacobian = []
        for i in range(outputs.shape[0]):
            # Zero out previous gradients
            NN.zero_grad()
            # Backward pass for the current output element
            outputs[i].backward(retain_graph=True)
            # Extract gradients and form a row of the Jacobian
            gradients = []
            for name, param in NN.named_parameters():  # models are the same for all dofs
                gradients.extend([param.grad.flatten()])
            jacobian.append(torch.cat(gradients).view(1, -1))
        return torch.cat(jacobian)
    
    def get_u(self, yref: Array, is_gradient: bool=False) -> Array:
        """
        """
        y_processed = self.DATA_PROCESS.get_data(raw_inputs=yref[0, 1:])
        y_tensor = torch.cat(y_processed, dim=0)
    
        if is_gradient is True:
            self.model.NN.train()
            u_tensor = self.model.NN(y_tensor.float())
            par_pi_par_omega = self.get_par_pi_par_omega(self.model.NN, u_tensor)
        else:
            self.model.NN.eval()
            u_tensor = self.model.NN(y_tensor.float())
            par_pi_par_omega = None
        
        u = self.tensor2np(u_tensor)
        return self.DATA_PROCESS.inverse_output(u), par_pi_par_omega

    def _rum_sim(self, env: environmnet, 
                 yref: Array2D, is_gradient: bool=False) -> Tuple:
        """
        """
        u, par_pi_par_omega = self.get_u(yref, is_gradient)
        yout, _ = env.one_step(u.flatten())
        loss = self.get_loss(yout.flatten(), yref[0, 1:].flatten())
        return yout, u, par_pi_par_omega, loss

    def online_learning(self, nr_iterations: int=100, 
                        is_shift_dis: bool=False,
                        is_clear: bool=False,
                        is_reset: bool=False) -> None:
        """
        """        
        self._online_learning(nr_iterations, 
                              is_shift_dis, 
                              is_clear,
                              is_reset)

    def shift_distribution(self, distribution: str='v1'):
        """change the distribution
        """
        self.traj_initialization(distribution)
        yref_marker, _ = self.traj.get_traj()
        return yref_marker

    def marker_initialization(self) -> Tuple[Array2D, Path]:
        """Generate the marker trajectory and the 
        path to marker folder
        """
        self.nr_marker = 0
        self.loss_marker = []
        self.total_loss = 0
        yref_marker, _ = self.traj.get_traj()
        path_marker = os.path.join(self.path_model, 'loss_marker')
        fcs.mkdir(path_marker)
        return yref_marker, path_marker

    def run_marker_step(self, env: environmnet,
                        yref: Array2D, path: Path) -> None:
        """Evaluate the marker trajectory
        """
        self.nr_marker += 1
        yout, u, _, loss = self._rum_sim(env, yref, is_gradient=False)
        self.loss_marker.append(np.round(loss, 7))
        fcs.print_info(
        Marker=[str(self.nr_marker)],
        Loss=[self.loss_marker[-6:]])

        path_marker_file = os.path.join(path, str(self.nr_marker))
        with open(path_marker_file, 'wb') as file:
            pickle.dump(yref, file)
            pickle.dump(yout, file)
            pickle.dump(u, file)
            pickle.dump(loss, file)

    def NN_update(self, NN: torch.nn, omega: torch.Tensor) -> None:
        """Update the parameters of the neural network
        """
        _omega = omega.clone()
        i = 0
        for name, param in NN.named_parameters():
            idx1 = self.nn_idx[i]
            idx2 = self.nn_idx[i+1]
            param.data = _omega[idx1:idx2].view(self.nn_shapes[i])
            i += 1
    
    def get_NN_params(self, NN: torch.nn) -> torch.Tensor:
        """Extract all the parameters of a neural network
        """
        self.nn_names = []
        self.nn_shapes = []
        idx = 0
        self.nn_idx = [idx]
        for name, param in NN.named_parameters():  # models are the same for all dofs
            self.nn_names.append(name)
            self.nn_shapes.append(param.shape)
            idx += len(param.data.view(-1))
            self.nn_idx.append(idx)

    def extract_parameters(self, NN: torch.nn) -> torch.Tensor:
        """Extract all the parameters of the neural network
        """
        return torch.cat([p.view(-1) for p in NN.parameters()])

    def flip_coin(self):
        return random.random() < 0.9
    
    def discrepancy_dectection(self, env: environmnet) -> Tuple[Array, list]:
        """
        """
        ydec, _ = self.traj.get_traj()
        yout_list = []
        for omega in self.online_optimizer.omega_list:
            self.NN_update(self.model.NN, omega)
            yout, _, _, _ = self._rum_sim(env, ydec)
            yout_list.append(yout)
        return ydec, yout_list
    
    @staticmethod
    def get_model_idx(n: int, m: int) -> int:
        """
        """
        while True:
            num = random.randint(0, n-1)
            if num != m:
                break
        return num
    
    def _online_learning(self, nr_iterations: int=100, 
                         is_shift_dis: bool=False,
                         is_clear: bool=False,
                         is_reset: bool=False):
        """Online learning using quasi newton method
        """
        self.get_NN_params(self.model.NN)
        omega = self.extract_parameters(self.model.NN)
        self.online_optimizer.ini_matrix(len(omega))
        self.online_optimizer.import_omega(omega)
        self.online_optimizer.save_latest_omega()

        yref_marker, path_marker = self.marker_initialization()

        model_idx = 0
        # model_switch_idx = [1, 2, 3, 4, 5]
        # model_switch_idx = [1000, 2000, 3000, 4000, 5000, 
        #                     6000, 7000, 8000, 9000, 10000, 
        #                     11000, 12000, 13000, 14000]
        model_switch_idx = [2000, 4000, 6000, 8000, 10000, 12000]
        
        for i in range(nr_iterations):
            tt = time.time()
            
            # if (is_shift_dis is True) and (i > self.nr_shift_dis):
            #     is_shift_dis = False
            #     yref_marker = self.shift_distribution()
                
            #     if is_clear is True:
            #         is_clear = False
            #         self.online_optimizer.clear_A()
                
            #     if is_reset is True:
            #         is_reset = False
            #         self.online_optimizer.import_omega(omega)
   
            if i in model_switch_idx:
                model_idx = 1 - model_idx
            #     model_idx = self.get_model_idx(len(self.envs), model_idx)

            #     self.online_optimizer.save_latest_omega()
            #     ydec, yout_list = self.discrepancy_dectection(self.envs[model_idx])
            #     self.online_optimizer.initialize_omega(ydec[0, 1:], yout_list)
            #     self.online_optimizer.clear_A()

            self.NN_update(self.model.NN, self.online_optimizer.omega)

            if i%self.nr_marker_interval == 0:
                self.run_marker_step(self.envs[model_idx], 
                                     yref_marker, path_marker)
            
            yref, _ = self.traj.get_traj()
   
            t1 = time.time()
            yout, u, par_pi_par_omega, loss = self._rum_sim(self.envs[model_idx], 
                                                            yref, is_gradient=True)
            tsim = time.time() - t1
            self.online_optimizer.import_par_pi_par_omega(par_pi_par_omega)
            self.online_optimizer.optimize(yref[0, 1:], yout)

            ttotal = time.time() - tt
            self.total_loss += loss
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Loss=[loss],
                AvgLoss=[self.total_loss/(i+1)],
                Ttotal=[ttotal],
                Tsim=[tsim],
                Model=[model_idx])
            
            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i,
                               u=u,
                               yref=yref,
                               yout=yout,
                               loss=loss,
                               gradient=self.online_optimizer.gradient,
                               model_idx=model_idx)
                
            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)


            

    


        