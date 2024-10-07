"""Classes for the online optimizer
"""
import numpy as np
import torch

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
    def __init__(self, delta: float, length: int, 
                 B: Array2D,
                 alpha: float, epsilon: float, 
                 rolling: int=20) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nr_iteration = 0
        self.delta = delta
        self.length = length
        self.alpha = alpha
        self.epsilon = epsilon
        self.rolling = rolling
        self.B = B
        self.Lambda_list = []
        self.initialization()

    def initialization(self) -> None:
        self.y = np.zeros((1, self.length))
        self.B = self.move_to_device(self.B)
        self.A = self.move_to_device(np.zeros((self.length-1, self.length-1)))
        self.I = self.move_to_device(np.eye(self.length-1))

    def move_to_device(self, data: Array) -> torch.Tensor:
        """Move data to the device
        """ 
        return torch.from_numpy(data).to(self.device).float()

    def import_omega(self, data: torch.Tensor) -> None:
        self.omega = data.clone().view(-1, 1)

    def import_par_pi_par_y(self, data: torch.Tensor) -> None:
        """Import par_pi_par_omega
        """
        self.par_pi_par_y = data.clone()
    
    @staticmethod
    def get_L(B, par_pi_par_y) -> torch.Tensor:
        """Get the L matrix
        """
        return torch.matmul(B, par_pi_par_y)
    
    @staticmethod
    def get_Lambda(L: torch.Tensor, par_pi_par_y: torch.Tensor, 
                   I: torch.Tensor, alpha: float, epsilon: float) -> torch.Tensor:
        """Get the single pseudo Hessian matrix
        """
        return torch.matmul(L.t(), L) + alpha*torch.matmul(par_pi_par_y.t(), par_pi_par_y) + epsilon*I 

    def update_Lambda(self, B: torch.Tensor) -> None:
        """Update Lambda list
        """
        self.L = self.get_L(B, self.par_pi_par_y)
        self.Lambda_list.append(self.get_Lambda(self.L, self.par_pi_par_y, self.I, self.alpha, self.epsilon))
        if len(self.Lambda_list) > self.rolling:
            self.Lambda_list.pop(0)
        
    def update_A(self, B: torch.Tensor):
        """Update the pseudo Hessian matrix
        """
        self.update_Lambda(B)     
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

    def update_model(self) -> torch.Tensor:
        l = self.yref.shape[0]
        return self.B[0:l, 0:l]

    def update_y(self) -> None:
        dy = self.dy.squeeze().to('cpu').numpy()
        dy = np.insert(dy, 0, 0.0, axis=0).reshape(1, -1)
        y_new = self.y + dy
        norm = np.linalg.norm(y_new, ord=2) 

        if norm <= self.delta:
            self.y = y_new
        else:
            self.y = y_new * (self.delta / norm)

    def _attack_newton(self) -> None:
        """Optimize the parameters using newton method
        """
        self._B = self.update_model()
        self.update_A(self._B)
        self.eta = 25.0

        self.gradient = self.get_gradient(self.L, self.yref, self.yout)        
        self.dy = self.eta*torch.matmul(torch.linalg.inv(self.A), self.gradient)
        # self.dy = self.eta*self.gradient
        self.update_y()

    # def _optimize_gradient(self) -> None:
    #     """Optimize the parameters using gradient descent method
    #     """
    #     B = self.update_model()
    #     self.L = self.get_L(B, self.par_pi_par_omega)
    #     self.gradient = self.get_gradient(self.L, self.yref, self.yout)
    #     self.eta = self.step_size.get_eta(self.nr_iteration)
    #     self.omega -= self.eta*self.gradient

    def save_latest_omega(self) -> None:
        """Save the latest well-trained parameters, when
        the distribution shift detected
        """
        self.omega_list.append(self.omega.clone())

    @staticmethod
    def normalize(lst):
        total = sum(lst)
        if total == 0:
            normalized_lst = [0 for x in lst]
        else:
            normalized_lst = [x / total for x in lst]
        return normalized_lst

    def attack(self, yref: Array, yout: Array) -> Array:
        """Do the online optimization
        """
        self.nr_iteration += 1

        self.yref = self.move_to_device(yref.reshape(-1, 1))
        self.yout = self.move_to_device(yout.reshape(-1, 1))

        self._attack_newton()
