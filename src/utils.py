"""Some useful functions
"""
import numpy as np
import os
from pathlib import Path
from matplotlib.axes import Axes
from typing import Any, List, Tuple
from tabulate import tabulate
import shutil
import torch
from scipy.signal import butter, filtfilt, freqz

from mytypes import Array, Array2D

def mkdir(path: Path) -> None:
    """Check if the folder exists and create it if it does not exist.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def _set_axes_radius_2d(ax, origin, radius) -> None:
    x, y = origin
    ax.set_xlim([x - radius, x + radius])
    ax.set_ylim([y - radius, y + radius])
    
def set_axes_equal_2d(ax: Axes) -> None:
    """Set equal x, y axes
    """
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius_2d(ax, origin, radius)

def set_axes_format(ax: Axes, x_label: str, y_label: str) -> None:
    """Format the axes
    """
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid()

def preprocess_kwargs(**kwargs):
    """Project the key
    """
    replacement_rules = {
        "__slash__": "/",
        "__percent__": "%"
    }

    processed_kwargs = {}
    key_map = {}
    for key, value in kwargs.items():
        new_key = key
        
        for old, new in replacement_rules.items():
            new_key = new_key.replace(old, new)

        processed_kwargs[key] = value
        key_map[key] = new_key
    
    return processed_kwargs, key_map

def print_info(**kwargs):
    """Print information on the screen
    """
    processed_kwargs, key_map = preprocess_kwargs(**kwargs)
    columns = [key_map[key] for key in processed_kwargs.keys()]
    data = list(zip(*processed_kwargs.values()))
    table = tabulate(data, headers=columns, tablefmt="grid")
    print(table)

def get_parent_path(lvl: int=0):
    """Get the lvl-th parent path as root path.
    Return current file path when lvl is zero.
    Must be called under the same folder.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if lvl > 0:
        for _ in range(lvl):
            path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def copy_folder(src, dst):
    try:
        if os.path.isdir(src):
            folder_name = os.path.basename(os.path.normpath(src))
            dst_folder = os.path.join(dst, folder_name)
            shutil.copytree(src, dst_folder)
            print(f"Folder '{src}' successfully copied to '{dst_folder}'")
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"File '{src}' successfully copied to '{dst}'")
        else:
            print(f"Source '{src}' is neither a file nor a directory.")
    except FileExistsError:
        print(f"Error: Destination '{dst}' already exists.")
    except FileNotFoundError:
        print(f"Error: Source '{src}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
def load_model(path: Path) -> None:
    """Load the model parameters
    
    parameters:
    -----------
    params_path: path to the pre-trained parameters
    """
    return torch.load(path)

def get_flatten(y: Array2D) -> Array:
    """Flatten an array by columns
    """
    dim = len(y.shape)
    if dim == 2:
        return y.flatten(order='F')
    elif dim == 3:
        num_envs, _, _ = y.shape
        return y.transpose(0, 2, 1).reshape(num_envs, -1)
    
def get_unflatten(u: Array, channels: int) -> Array2D:
    """Unflatten the array
    """
    return u.reshape((channels, -1), order='F')

def add_one(a: Array) -> Array:
    """add element one
    """
    return np.hstack((a.flatten(), 1))

def adjust_matrix(matrix: Array2D, new_matrix: Array2D, 
                  max_rows: int) -> Array2D:
    """Vertically concatenate matrices, and if 
    the resulting number of rows exceeds the given 
    limit, remove rows from the top of the matrix.
    """
    if matrix is None:
        return new_matrix.copy()
    else:
        original_rows = matrix.shape[0]
        combined_matrix = np.vstack((matrix, new_matrix))

        if original_rows >= max_rows:
            combined_matrix = combined_matrix[-max_rows:, :]
    
        return combined_matrix
    
def diagonal_concatenate(A, B, max_size):
    """
    """
    size_A = A.shape[0]
    size_B = B.shape[0]
    
    result_size = size_A + size_B
    result = np.zeros((result_size, result_size))
    
    result[:size_A, :size_A] = A
    result[size_A:, size_A:] = B
    
    if result_size > max_size:
        result = result[size_B:, size_B:]
    
    return result

def _get_martingale(data, mode, rolling):
    num = len(data)
    rolling_list = []
    martingale_list = []
    for i in range(num):
        rolling_list.append(data[i].flatten())

        if mode == 'rolling':
            if len(rolling_list) > rolling:
                rolling_list.pop(0)
        
        sum_value = sum(rolling_list)
        martingale_list.append(np.linalg.norm(sum_value/len(rolling_list)))
    return martingale_list

def get_martingale(data_list, mode, rolling):
    if isinstance(data_list[0], list):
        num = len(data_list)
        martingale_list = []
        for i in range(num):
            martingale_list.append(_get_martingale(data_list[i], mode, rolling))
    elif isinstance(data_list[0], np.ndarray):
        martingale_list = _get_martingale(data_list, mode, rolling)
    return martingale_list

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def add_noise(y, snr_db=20):
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, size=y.shape)

    cutoff = 10 
    order = 4
    fs = 500
    b, a = butter_lowpass(cutoff, fs, order)
    filtered_noise = filtfilt(b, a, noise)

    filtered_noise[0, 0] = 0.0
    filtered_noise[0, -51:] = 0.0
    return y + noise
    # return y + filtered_noise