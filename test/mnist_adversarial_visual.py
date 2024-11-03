"""Test for mnist visual
"""
import os, sys
import torch
import numpy as np
import torch.nn.functional as F
import pickle
from scipy.special import softmax
import matplotlib.pyplot as plt

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

def list_files_os_listdir(directory):
    try:
        entries = os.listdir(directory)
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
        return files
    except FileNotFoundError:
        print(f"'{directory}' does not exist")
        return []

def _get_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['image'], data['dy']

def get_label(y):
    return np.argmax(y)

def get_unflatten_tensor(y):
    """
    """
    if y.numel() != 784:
        raise ValueError("Dim is wrong!")
    image = y.view(28, 28)
    image = image.t()
    image = image.unsqueeze(0).unsqueeze(0)
    return image

def get_data(path):
    images = []
    preds = []
    labels =[]
    dys = []
    distributions = []
    files = list_files_os_listdir(path)
    for i in range(len(files)):
        path_data = os.path.join(path, str(i))
        image, dy = _get_data(path_data)
        # pred = get_label(softmax(yout))
        # label = get_label(yref)
        images.append(image)
        dys.append(get_unflatten_tensor(dy))
        # preds.append(pred)
        # labels.append(label)
        # if i == 0:
        #     distributions.append(dis)
        # else:
        #     if np.any(dis != distributions[-1]):
        #         distributions.append(dis)

    return images, dys

def get_accuracy(preds, labels, l=500):
    nr = len(preds)
    nr_acc = 0
    acc = []
    nr_acc_l = 0
    for i in range(nr):
        if preds[i] == labels[i]:
            nr_acc += 1
            if i >= (nr-l):
                nr_acc_l += 1
        acc.append(nr_acc*100/(i+1))
    
    acc_l = nr_acc_l*100 / l
    return acc, acc_l

def get_plot(path, acc):
    path_save = os.path.join(path, 'accuracy.png')
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fcs.set_axes_format(ax, r'Epoch', r'Accuracy\%')
    ax.semilogy(acc, linewidth=0.5, linestyle='-')
    ax.legend(fontsize=14)
    plt.savefig(path_save)
    plt.show()

def _get_plot(img, path):
    img = img * 0.3081 + 0.1307 
    np_img = img.to('cpu').detach().numpy().squeeze()
    
    plt.imshow(np_img, cmap='gray')
    plt.axis('off')
    plt.savefig(path)
    # plt.show()

def get_plots(images, path):
    nr = len(images)
    for i in range(nr):
        if i%20 == 0:
            path_fig = os.path.join(path, str(i)+'.png')
            _get_plot(images[i], path_fig)

def test():
    """main script
    1. load the data
    2. calculate the accuracy
    """
    folder1 = 'mnist_adversarial_attack'
    folders = ['test']
    
    if len(folders) == 0:
        folders = get_all_folders(folder1)

    for folder in folders:
        path = os.path.join(root, 'data', folder1, folder)
        path_data = os.path.join(path, 'data')
        images, dys = get_data(path_data)

        path_fig = os.path.join(path, 'figure')
        fcs.mkdir(path_fig)
        get_plots(images, path_fig)
        path_fig = os.path.join(path, 'dy')
        fcs.mkdir(path_fig)
        get_plots(dys, path_fig)


if __name__ == '__main__':
    test()