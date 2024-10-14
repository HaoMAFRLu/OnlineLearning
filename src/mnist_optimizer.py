"""Classes for the online optimizer
"""
import numpy as np
import time
import torch

from mytypes import Array, Array2D
import utils as fcs
from step_size import StepSize

class MNISTOptimizer():
    """The class for online quais-Newton method

    parameters:
    -----------
    mode: gradient descent method or newton method
    B: identified linear model
    """
    def __init__(self, mode: str, alpha: float, epsilon: float, 
                 eta: float, gamma: float, rolling: int=50) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.alpha = alpha
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma
        self.nr_iteration = 0
        self.rolling = rolling
        self.Lambda_list = []
        self.omega_list = []
        self.step_size = StepSize('constant', {'value0': self.eta})

    def ini_matrix(self, dim: int) -> None:
        """Initialize the matrices
        """
        if self.mode == 'newton':
            self.A = self.move_to_device(np.zeros((dim, dim)))
            self.I = self.move_to_device(np.eye(dim))
        elif self.mode == 'gradient':
            self.A = []

    def move_to_device(self, data: Array) -> torch.Tensor:
        """Move data to the device
        """ 
        return torch.from_numpy(data).to(self.device).float()

    def import_omega(self, data: torch.Tensor) -> None:
        self.omega = data.clone().view(-1, 1)

    def import_par_pi_par_omega(self, data: torch.Tensor) -> None:
        """Import par_pi_par_omega
        """
        self.par_pi_par_omega = data.clone()
    
    @staticmethod
    def get_Lambda(L: torch.Tensor, par_pi_par_omega: torch.Tensor, I: torch.Tensor,
                   alpha: float, epsilon: float) -> torch.Tensor:
        """Get the single pseudo Hessian matrix
        """
        return torch.matmul(L.t(), L) + alpha*torch.matmul(par_pi_par_omega.t(), par_pi_par_omega) + epsilon*I 

    def update_Lambda(self) -> None:
        """Update Lambda list
        """
        self.Lambda_list.append(self.get_Lambda(self.par_pi_par_omega, self.par_pi_par_omega, self.I, self.alpha, self.epsilon))
        if len(self.Lambda_list) > self.rolling:
            self.Lambda_list.pop(0)
        
    def update_A(self):
        """Update the pseudo Hessian matrix
        """
        self.update_Lambda()     
        self.A = sum(self.Lambda_list)/len(self.Lambda_list)

    @staticmethod
    def get_gradient(L: torch.Tensor, 
                     yref: torch.Tensor, yout: torch.Tensor) -> torch.Tensor:
        """
        """
        return torch.matmul(L.t(), yout - yref)

    def clear_A(self) -> None:
        self.A = self.A*0.0
        self.Lambda_list = []

    def _optimize_newton(self) -> None:
        """Optimize the parameters using newton method
        """
        self.update_A()
        self.eta = self.step_size.get_eta(self.nr_iteration)

        self.gradient = self.get_gradient(self.par_pi_par_omega, self.yref, self.yout)        
        self.omega -= self.eta*torch.matmul(torch.linalg.inv(self.A), self.gradient)

    def _optimize_gradient(self) -> None:
        """Optimize the parameters using gradient descent method
        """
        self.gradient = self.get_gradient(self.par_pi_par_omega, self.yref, self.yout)
        self.eta = self.step_size.get_eta(self.nr_iteration)
        self.omega -= self.eta*self.gradient

    def optimize(self, yref: Array, yout: Array) -> Array:
        """Do the online optimization
        """
        self.nr_iteration += 1

        self.yref = self.move_to_device(yref.reshape(-1, 1))
        self.yout = self.move_to_device(yout.reshape(-1, 1))

        if self.mode == 'gradient':
            self._optimize_gradient()
        elif self.mode == 'newton':
            self._optimize_newton()
