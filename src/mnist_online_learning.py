"""Classes for online learning for mnist dataset
"""

"""Classes for online learning algorithm
"""
import torch
from pathlib import Path
import os, sys
from typing import Tuple, List
import pickle
import numpy as np
from datetime import datetime
import time
from scipy.signal import butter, filtfilt, freqz
import random
from scipy.special import softmax

random.seed(10086)

import utils as fcs
from mytypes import Array, Array2D, Array3D

from mnist_optimizer import MNISTOptimizer
import networks
from mnist_generator import MNISTGenerator
import params

class MNISTOnlineLearning():
    """Classes for online learning
    """
    def __init__(self, mode: str='gradient',
                 nr_interval: int=5000,
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
    
    def NN_initialization(self, path: Path, PARAMS: dict) -> None:
        """Build the model and load the pretrained weights
        """
        self.model = networks.MNIST_CNN(self.device, PARAMS)
        self.model.build_network()
        
        if path is not None:
            checkpoint = torch.load(path)
            self.model.NN.load_state_dict(checkpoint['model_state_dict'])

    def data_generator_initialization(self) -> None:
        """Create the generator for the images and labels
        """
        self.data_generator = MNISTGenerator('train')
    
    def mnist_optimizer_initialization(self) -> None:
        """Initialize the kalman filter
        """
        self.mnist_optimizer = MNISTOptimizer(mode=self.mode, alpha=self.alpha, epsilon=self.epsilon,
                                               eta=self.eta, gamma=self.gamma)

    @staticmethod
    def get_params(path: Path) -> Tuple[dict]:
        """Return the hyperparameters for each module

        parameters:
        -----------
        path: path to folder of the config file
        """
        PATH_CONFIG = os.path.join(path, 'config.json')
        PARAMS_LIST = ["NN_PARAMS"]
        params_generator = params.PARAMS_GENERATOR(PATH_CONFIG)
        params_generator.get_params(PARAMS_LIST)
        return (params_generator.PARAMS['NN_PARAMS'])

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
        NN_PARAMS = self.get_params(self.root)
        self.data_generator_initialization()
        self.NN_initialization(None, NN_PARAMS)
        self.mnist_optimizer_initialization()
    
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
        for i in range(outputs.shape[1]):
            # Zero out previous gradients
            NN.zero_grad()
            # Backward pass for the current output element
            outputs[0, i].backward(retain_graph=True)
            # Extract gradients and form a row of the Jacobian
            gradients = []
            for name, param in NN.named_parameters():  # models are the same for all dofs
                gradients.extend([param.grad.flatten()])
            jacobian.append(torch.cat(gradients).view(1, -1))
        return torch.cat(jacobian)
    
    def classification(self, model, data, is_gradient=False):
        """
        """
        if is_gradient is True:
            self.model.NN.train()
            yout = model.NN(data)
            par_pi_par_omega = self.get_par_pi_par_omega(self.model.NN, yout)
        else:
            self.model.NN.eval()
            yout = model.NN(data)
            par_pi_par_omega = None
        return self.tensor2np(yout), par_pi_par_omega
        
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

    def get_distribution(self) -> list:
        """
        """
        random_numbers = np.random.dirichlet(np.ones(10))
        return list(random_numbers)

    def shift_distribution(self) -> None:
        """change the distribution
        """
        distribution = self.get_distribution()
        self.data_generator.update_distribution(distribution)
        
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

    def get_one_hot(self, label, nr_classes: int=10):
        """Return one-hot
        """
        label = np.array(label)
        return np.eye(nr_classes)[label]
    
    def _online_learning(self, nr_iterations: int=100, 
                         is_shift_dis: bool=False,
                         is_clear: bool=False,
                         is_reset: bool=False):
        """Online learning using quasi newton method
        """
        self.get_NN_params(self.model.NN)
        omega = self.extract_parameters(self.model.NN)
        self.mnist_optimizer.ini_matrix(len(omega))
        self.mnist_optimizer.import_omega(omega)

        dis_switch_idx = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]
        total_loss = 0.0
        nr_acc = 0

        for i in range(nr_iterations):
            tt = time.time()
            
            if (is_shift_dis is True) and (i in dis_switch_idx):
                self.shift_distribution()

            #     is_shift_dis = False
            #     yref_marker = self.shift_distribution()
                
            #     if is_clear is True:
            #         is_clear = False
            #         self.online_optimizer.clear_A()
                
            #     if is_reset is True:
            #         is_reset = False
            #         self.online_optimizer.import_omega(omega)
   
            # if i in model_switch_idx:
            #     model_idx = 1 - model_idx
            #     model_idx = self.get_model_idx(len(self.envs), model_idx)

            #     self.online_optimizer.save_latest_omega()
            #     ydec, yout_list = self.discrepancy_dectection(self.envs[model_idx])
            #     self.online_optimizer.initialize_omega(ydec[0, 1:], yout_list)
            #     self.online_optimizer.clear_A()

            self.NN_update(self.model.NN, self.mnist_optimizer.omega)
   
            data, label = self.data_generator.get_samples()
            data = data.to(self.device)
            yref = self.get_one_hot(label)

            t1 = time.time()
            yout, par_pi_par_omega = self.classification(self.model, data, is_gradient=True)
            ysoftmax = softmax(yout)
            tsim = time.time() - t1
            self.mnist_optimizer.import_par_pi_par_omega(par_pi_par_omega)
            self.mnist_optimizer.optimize(yref, ysoftmax)
            
            if label.numpy() == np.argmax(ysoftmax):
                nr_acc += 1

            loss = self.get_loss(yref, yout)
            total_loss += loss

            ttotal = time.time() - tt
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                Acc=[str((nr_acc/(i+1))*100) + '%'],
                Label=[label],
                Pred=[np.argmax(ysoftmax)],
                Ttotal=[ttotal],
                Tsim=[tsim])
            
            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i,
                               data=data,
                               yref=yref,
                               yout=yout,
                               distribution=self.data_generator.distribution)
                
            if (i+1) % self.nr_interval == 0:
                self.save_checkpoint(i+1)


            

    


        