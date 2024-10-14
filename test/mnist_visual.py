"""Test for mnist visual
"""
import os, sys
import torch
import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import utils as fcs
from mnist_optimizer import MNISTOptimizer
import networks
from mnist_generator import MNISTGenerator
import params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = fcs.get_parent_path(lvl=1)
    
def build_model(PARAMS):
    """Build the model
    """
    model = networks.MNIST_CNN(device, PARAMS)
    model.build_network()
    return model

def find_pth_files_os(root_dir):
    pth_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
    return pth_files

def load_model(model, path):
    """Load the pretrained model
    """
    files = find_pth_files_os(path)
    nr_models = len(files)
    if nr_models > 0:
        model_name = 'checkpoint_epoch_' + str(nr_models*1000) + '.pth'
        path_model = os.path.join(path, model_name)
        checkpoint = torch.load(path_model)
        model.NN.load_state_dict(checkpoint['model_state_dict'])
        return True
    else:
        return False

def get_params():
    """Return the hyperparameters for each module

    parameters:
    -----------
    path: path to folder of the config file
    """
    PATH_CONFIG = os.path.join(root, 'src', 'config.json')
    PARAMS_LIST = ["NN_PARAMS"]
    params_generator = params.PARAMS_GENERATOR(PATH_CONFIG)
    params_generator.get_params(PARAMS_LIST)
    return (params_generator.PARAMS['NN_PARAMS'])

def data_generator_initialization():
    """Create the class of reference trajectories
    """
    data_generator = MNISTGenerator('test')
    return data_generator

def initialization():
    NN_PARAMS = get_params()
    data_generator = data_generator_initialization()
    return NN_PARAMS, data_generator

def tensor2np(a: torch.tensor):
    """Covnert tensor to numpy
    """
    return a.squeeze().to('cpu').detach().numpy()

def get_loss(y1, y2):
    return np.linalg.norm(y1-y2)

def list_directories(path):
    entries = os.listdir(path)
    directories = [entry for entry in entries 
                   if os.path.isdir(os.path.join(path, entry)) 
                   and entry.lower() != 'raw'
                   and not entry.startswith('.')]
    return directories

def get_all_folders(folder):
    path = os.path.join(root, 'data', folder)
    return list_directories(path)

def get_accuracy(yout, labels):
    softmaxed = F.softmax(yout, dim=1)
    max_indices = torch.argmax(softmaxed, dim=1).to('cpu')
    correct = (labels == max_indices).sum().item()
    total = labels.size(0)
    accuracy = correct / total * 100
    return accuracy

def run_model(model, images, labels, path) -> float:
    is_load = load_model(model, path)
    if is_load is True:
        model.NN.eval()
        yout = model.NN(images)
        accuracy = get_accuracy(yout, labels)
    else:
        accuracy = 0.0
    return accuracy

def list_files_os_listdir(directory):
    try:
        entries = os.listdir(directory)
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
        return files
    except FileNotFoundError:
        print(f"'{directory}' does not exist")
        return []

def get_nr_iter(path):
    path_data = os.path.join(path, 'data')
    files = list_files_os_listdir(path_data)
    return len(files)

def get_data(data_generator):
    images, labels = data_generator.get_samples()
    images = images.to(device)
    return images, labels

def get_hyperparameters(folder):
    parts = folder.split('_')
    alpha = float(parts[0])
    epsilon = float(parts[1])
    eta = float(parts[2])
    return alpha, epsilon, eta

def get_print(rank, acc, folder, nr_iter):
    alpha, epsilon, eta = get_hyperparameters(folder)
    fcs.print_info(Rank=[rank+1],
                Acc=[str(acc) + '%'],
                alpha=[alpha],
                epsilon=[epsilon],
                eta=[eta],
                Nr_Iter=[nr_iter])

def test():
    """main script
    1. load the neural network
    2. initialize the data generator
    3. test the accuracy
    """
    folder1 = 'mnist_wo_shift2'
    nr_to_print = 100
    folders = []
    acc_list = []
    iter_list = []
    
    if len(folders) == 0:
        folders = get_all_folders(folder1)

    NN_PARAMS, data_generator = initialization()
    model = build_model(NN_PARAMS)
    images, labels = get_data(data_generator)

    for folder in folders:
        path = os.path.join(root, 'data', folder1, folder)
        nr_iter = get_nr_iter(path)
        accuracy = run_model(model, images, labels, path)
        acc_list.append(accuracy)
        iter_list.append(nr_iter)
    
    combined = zip(acc_list, folders, iter_list)
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    sorted_acc, sorted_folders, sorted_nr_iter = zip(*sorted_combined)

    for i in range(nr_to_print):
        get_print(i, sorted_acc[i], sorted_folders[i], sorted_nr_iter[i])
if __name__ == '__main__':
    test()