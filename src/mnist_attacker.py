"""Classes for the online optimizer
"""
import numpy as np
import torch
import torch.nn.functional as F

from mytypes import Array, Array2D
import utils as fcs
from step_size import StepSize

class Attacker():
    """The class for online quais-Newton method

    parameters:
    -----------
    mode: gradient descent method or newton method
    B: identified linear model
    """
    def __init__(self, mode: str,
                 delta: float, label: int,
                 alpha: float, epsilon: float, eta: float,
                 rolling: int=1) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.nr_iteration = 0
        self.delta = delta
        self.length = 28*28
        self.eta = eta
        self.label = label
        self.alpha = alpha
        self.epsilon = epsilon
        self.rolling = rolling
        self.Lambda_list = []
        self.initialization()

    def initialization(self) -> None:
        self.y = self.move_to_device(np.zeros((self.length, 1)))
        self.A = self.move_to_device(np.zeros((self.length, self.length)))
        self.I = self.move_to_device(np.eye(self.length))

    def move_to_device(self, data: Array) -> torch.Tensor:
        """Move data to the device
        """ 
        return torch.from_numpy(data).to(self.device).float()

    def import_par_pi_par_y(self, data: torch.Tensor) -> None:
        """Import par_pi_par_omega
        """
        self.par_pi_par_y = data.clone()
    
    @staticmethod
    def get_Lambda(L: torch.Tensor, par_pi_par_y: torch.Tensor, 
                   I: torch.Tensor, alpha: float, epsilon: float) -> torch.Tensor:
        """Get the single pseudo Hessian matrix
        """
        return torch.matmul(L.t(), L) + alpha*torch.matmul(par_pi_par_y.t(), par_pi_par_y) + epsilon*I 

    def update_Lambda(self) -> None:
        """Update Lambda list
        """
        self.Lambda_list.append(self.get_Lambda(self.par_pi_par_y, self.par_pi_par_y, self.I, self.alpha, self.epsilon))
        if len(self.Lambda_list) > self.rolling:
            self.Lambda_list.pop(0)
        
    def update_A(self):
        """Update the pseudo Hessian matrix
        """
        self.update_Lambda()     
        self.A = sum(self.Lambda_list)/len(self.Lambda_list)

    @staticmethod
    def get_gradient(L: torch.Tensor, par_y) -> torch.Tensor:
        """
        """
        return torch.matmul(L.t(), par_y)

    def clear_A(self) -> None:
        self.A = self.A*0.0
        self.Lambda_list = []

    def update_y(self) -> None:
        y_new = self.y + self.dy
        norm = torch.linalg.norm(y_new, ord=2) 

        # if norm <= self.delta:
        if 1:
            self.y = y_new
        else:
            self.y = y_new * (self.delta / norm)

    def _attack_newton(self) -> None:
        """Optimize the parameters using newton method
        """
        
        self.gradient = self.get_gradient(self.par_pi_par_y, self.par_y)        
        if self.mode == 'newton':
            self.update_A()
            self.dy = self.eta*torch.matmul(torch.linalg.inv(self.A), self.gradient)
        elif self.mode == 'gradient':
            self.dy = self.eta*self.gradient
        self.update_y()

    @staticmethod
    def normalize(lst):
        total = sum(lst)
        if total == 0:
            normalized_lst = [0 for x in lst]
        else:
            normalized_lst = [x / total for x in lst]
        return normalized_lst

    def get_par_y(self, yout):
        """
        """
        ysoftmax = F.softmax(yout, dim=1).reshape(-1, 1)
        self.par_y = -ysoftmax*ysoftmax[self.label]
        self.par_y[self.label] = ysoftmax[self.label] + self.par_y[self.label]

    def attack(self, yout) -> Array:
        """Do the online optimization
        """
        self.nr_iteration += 1
        self.get_par_y(yout)
        self._attack_newton()
