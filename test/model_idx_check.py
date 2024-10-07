"""Collect variables
"""
import os, sys
import pickle
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

def is_folder_empty(folder_path):
    contents = os.listdir(folder_path)
    return len(contents) == 0

def list_files_in_folder(folder_path):
    contents = os.listdir(folder_path)
    files = [f for f in contents if os.path.isfile(os.path.join(folder_path, f))]
    return files

def read_data(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)
        # data = torch.load(file, map_location=torch.device('cpu'))
    return data

def read_model_idx(path):
    folder = os.path.join(path, 'data')
    files = list_files_in_folder(folder)
    files = sorted(files, key=int)
    
    model_idx = []
    for i in files:
        path_file = os.path.join(folder, i)
        data = read_data(path_file)
        model_idx.append(data['model_idx'])

    return model_idx

def remove_consecutive_duplicates(lst):
    if not lst:
        return []
    result = [lst[0]] 
    for item in lst[1:]:
        if item != result[-1]:
            result.append(item)
    return result

if __name__ == '__main__':
    root = fcs.get_parent_path(lvl=1)
    folder1 = 'multi_dynamics'
    folder = '0.1_5.0_25.0_0.5'
    path = os.path.join(root, 'data', folder1, folder)
    
    model_idx = read_model_idx(path)
    idx = remove_consecutive_duplicates(model_idx)
    print(idx)
