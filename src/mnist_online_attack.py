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

import params
from mnist_attacker import Attacker
from mnist_generator import MNISTGenerator
import networks

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
        self.mode = mode
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
        PARAMS_LIST = ["NN_PARAMS"]
        params_generator = params.PARAMS_GENERATOR(PATH_CONFIG)
        params_generator.get_params(PARAMS_LIST)
        return (params_generator.PARAMS['NN_PARAMS'])

    def NN_initialization(self, PARAMS: dict) -> None:
        """Build the model and load the pretrained weights
        """
        self.model = networks.MNIST_CNN(self.device, PARAMS)
        self.model.build_network()

    def online_attacker_initialization(self, delta: float, label: int,
                                       alpha: float, epsilon: float, eta: float) -> None:
        """Initialize the kalman filter
        """
        self.online_attacker = Attacker(self.mode, delta=delta, label=label, alpha=alpha, 
                                        epsilon=epsilon, eta=eta)

    def data_generator_initialization(self) -> None:
        """Create the generator for the images and labels
        """
        self.data_generator = MNISTGenerator('train')

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
    
    def online_adversarial_attack(self, nr_iterations: int=100, delta: float=1.0,
                                  alpha: float=0.1, epsilon: float=0.1,
                                  eta: float=0.1) -> None:
        """
        """        
        self._online_attack(nr_iterations, delta, alpha, epsilon, eta)

    @staticmethod
    def get_unflatten_tensor(y):
        """
        """
        if y.numel() != 784:
            raise ValueError("Dim is wrong!")
        image = y.view(28, 28)
        image = image.t()
        image = image.unsqueeze(0).unsqueeze(0)
        return image
    
    def image_update(self, image: Array2D, dy: Array2D) -> None:
        """Update the parameters of the neural network
        """
        dy_unflatten = self.get_unflatten_tensor(dy.flatten())
        return image.clone() + dy_unflatten
    
    def get_energy(self, d) -> float:
        return np.linalg.norm(d.to('cpu').detach().numpy())
    
    def get_flatten(self, image):
        """
        """
        image = image.squeeze()
        image = image.transpose(0, 1)
        tensor = image.flatten()
        return tensor

    def get_par_pi_par_y(self, NN, image, yout):
        nr_classes = 10
        gradients = []
        for i in range(nr_classes):
            NN.zero_grad()
            if image.grad is not None:
                image.grad.zero_()

            selected_output = yout[0, i]
            selected_output.backward(retain_graph=True)  # 保留计算图以便多次反向传播
            
            grad = image.grad.clone()
            gradients.append(self.get_flatten(grad))
    
            image.grad.zero_()

        gradients = torch.stack(gradients, dim=0)
        return gradients

    def classification(self, NN, image, is_gradient=False):
        """
        """
        if is_gradient is True:
            NN.train()
            image = image.clone().detach().requires_grad_(True)
            yout = NN(image)
            par_pi_par_y = self.get_par_pi_par_y(NN, image, yout)
        else:
            NN.eval()
            yout = NN(image)
            par_pi_par_y = None
        return yout, par_pi_par_y

    # def matrix_initialization(self, l: int, hl: int=100, hr: int=100) -> None:
    #     Z = np.zeros((hl+hl+1, l+hl+hr))
    #     I = np.eye(hl+hr+1)
    #     par_pi_par_y = np.zeros((l, l))
    #     self.par_pi_par_y = torch.from_numpy(par_pi_par_y).to(self.device).float()
    #     self.Z = torch.from_numpy(Z).to(self.device).float()
    #     self.I = torch.from_numpy(I).to(self.device).float()

    def _online_attack(self, nr_iterations: int, delta: float,
                       alpha: float, epsilon: float, eta: float):
        """Pipeline for online adversarial attack using gradient information
        """
        image, label = self.data_generator.get_samples()
        image, label = self.data_generator.get_samples()

        label = label.item()
        label = 2
        image = image.to(self.device)
        image_ini = image.clone()

        self.online_attacker_initialization(delta, label, alpha, epsilon, eta)
        # self.matrix_initialization(image)

        for i in range(nr_iterations):
            tt = time.time()

            image = self.image_update(image_ini, self.online_attacker.y)

            t1 = time.time()
            yout, par_pi_par_y = self.classification(self.model.NN, image, is_gradient=True)
            input_energy = self.get_energy(self.online_attacker.y)
            pred = torch.argmax(yout).item()
            tsim = time.time() - t1
            
            self.online_attacker.import_par_pi_par_y(par_pi_par_y)
            self.online_attacker.attack(yout)

            ttotal = time.time() - tt
            fcs.print_info(
                Epoch=[str(i+1)+'/'+str(nr_iterations)],
                InputEnergy=[input_energy],
                Label=[label],
                Pred=[pred],
                Ttotal=[ttotal],
                Tsim=[tsim])
            
            if (i+1) % self.nr_data_interval == 0:
                self.save_data(i,
                               image=image,
                               label=label,
                               pred=pred,
                               yout=yout,
                               dy=self.online_attacker.y)
                
            # if (i+1) % self.nr_interval == 0:
            #     self.save_checkpoint(i+1)


            

    


        