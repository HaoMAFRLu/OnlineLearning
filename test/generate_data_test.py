"""Test for class DataProcess
"""
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_process import DataProcess
from params import PARAMS_GENERATOR
from trajectory import TRAJ

def data_test():
    traj_generator = TRAJ()
    yref, _ = traj_generator.get_traj()

    params_generator = PARAMS_GENERATOR()
    params_generator.get_params('DATA_PARAMS')
    DATA_PROCESS = DataProcess('online', 
                               params_generator.PARAMS['DATA_PARAMS'])
    data = DATA_PROCESS.get_data(raw_inputs=yref)
    print('here')

if __name__ == "__main__":
    data_test()