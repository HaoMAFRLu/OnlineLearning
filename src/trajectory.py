"""Classes for reference trajectories
"""
import numpy as np
import random

random.seed(9527)

import minimum_jerk.minjerk as mj
from mytypes import Array, Array2D

class TRAJ():
    """
    """
    def __init__(self, distribution: str='original') -> None:
        self.T       = 5.5
        self.dt      = 0.01
        self.range_y = 0.2
        self.range_v = 2.0
        self.distribution = distribution
        if self.distribution == 'v1':
            self.nr_points = random.randint(1, 8)

    def get_random_value(self, start: float, 
                         end: float, step: float) -> float:
        """Randomly choose one value in [start, end] with interval step
        """
        values = np.arange(start, end, step)
        random_value = random.choice(values)
        return random_value

    def _get_t(self) -> Array:
        t_stamp = [0.0]
        for i in range(self.nr_points):
            _t = t_stamp[i]
            l = _t + 0.8
            r = _t + 1.4
            t_stamp.append(self.get_random_value(l, r, self.dt))
        t_stamp.append(t_stamp[-1]+0.5)
        return np.array(t_stamp)
    
    def _get_y(self) -> Array:
        y_stamp = [0.0]
        for i in range(self.nr_points):
            y_stamp.append(self.get_random_value(-self.range_y, self.range_y, self.dt))
        y_stamp.append(y_stamp[-1])
        return np.array(y_stamp)    

    def _get_v(self) -> Array:
        v_stamp = [0.0]
        for i in range(self.nr_points-1):
            v_stamp.append(self.get_random_value(-self.range_v, self.range_v, self.dt))
        v_stamp.append(0.0)
        v_stamp.append(0.0)
        return np.array(v_stamp)

    def get_t(self) -> Array:
        """Get the array of time points
        """
        if self.distribution == 'original':
            return np.array([0.0,  self.get_random_value(1.2, 1.8, self.dt),
                            self.get_random_value(2.9, 3.5, self.dt), 5.0, self.T])
        elif self.distribution == 'tmp':
            return np.array([0.0,  self.get_random_value(1.1, 1.5, self.dt),
                             self.get_random_value(2.7, 3.2, self.dt), 5.0, self.T])
        elif self.distribution == 'shift':
            return np.array([0.0,  self.get_random_value(0.8, 1.4, self.dt),
                             self.get_random_value(2.0, 2.6, self.dt),
                             self.get_random_value(3.2, 3.8, self.dt), 5.0, self.T])
        elif self.distribution == 'v1':
            return self._get_t()

    def get_y(self) -> Array:
        """Get the array of the positions
        """
        if self.distribution == 'original':
            return np.array([0.0, self.get_random_value(-self.range_y, self.range_y, self.dt),
                            self.get_random_value(-self.range_y, self.range_y, self.dt), 0.0, 0.0])
        elif self.distribution == 'tmp':
            return np.array([0.0, self.get_random_value(-0.4, 0.0, self.dt),
                             self.get_random_value(0.0, 0.4, self.dt), 0.0, 0.0])
        elif self.distribution == 'shift':
            return np.array([0.0, self.get_random_value(-0.8, -0.4, self.dt),
                             self.get_random_value(-0.5, 0.5, self.dt),
                             self.get_random_value(0.4, 0.8, self.dt), 0.0, 0.0])
        elif self.distribution == 'v1':
            return self._get_y()

    def get_v(self) -> Array:
        """Get the array of the velocities
        """
        if self.distribution == 'original':
            return np.array([0.0, self.get_random_value(-self.range_v, self.range_v, self.dt),
                            self.get_random_value(-self.range_v, self.range_v, self.dt), 0.0, 0.0])
        elif self.distribution == 'tmp':
            return np.array([0.0, self.get_random_value(-1.5, 1.5, self.dt),
                             self.get_random_value(-1.5, 1.5, self.dt), 0.0, 0.0])
        elif self.distribution == 'shift':
            return np.array([0.0, self.get_random_value(-1.0, 1.0, self.dt),
                             self.get_random_value(-2.0, 2.0, self.dt),
                             self.get_random_value(-1.0, 1.0, self.dt), 0.0, 0.0])
        elif self.distribution == 'v1':
            return self._get_v()
    
    def get_a(self) -> Array:
        """Get the array of the accelerations
        """
        if self.distribution == 'original':
            return np.zeros(5)
        elif self.distribution == 'tmp':
            return np.zeros(5)
        elif self.distribution == 'shift':
            return np.zeros(6)
        elif self.distribution == 'v1':
            return np.zeros(self.nr_points+2)
        
    def get_traj(self) -> Array:
        """
        """
        t = self.get_t()
        y = self.get_y()
        v = self.get_v()
        a = self.get_a()
        pp, vv, aa, jj, tt = mj.minimum_jerk_trajectory(y, v, a, t, self.dt)  
        return pp, tt