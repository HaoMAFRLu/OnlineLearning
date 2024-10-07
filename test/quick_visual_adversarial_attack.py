import pickle
import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

def list_files(directory):
    items = os.listdir(directory)
    files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
    return files

def test():
    is_save = True

    root = fcs.get_parent_path(lvl=1)
    folder = "test"
    file = "20241007_093127"
    path = os.path.join(root, 'data', folder, file)
    path_data = os.path.join(path, 'data')
    path_figure = os.path.join(path, 'figure')
    fcs.mkdir(path_figure)

    files = list_files(path_data)

    loss_list = []
    path_train_fig = os.path.join(path_figure, 'train')
    fcs.mkdir(path_train_fig)

    for i in range(len(files)):
        path_file = os.path.join(path_data, str(i))
        with open(path_file, 'rb') as file:
            data = pickle.load(file)
        
        yout = data["yout"].flatten()
        yref = data["yref"].flatten()[1:]
        u = data["u"].flatten()
        dy = data["dy"]

        if i == 0:
            yref_ini = yref.copy()
            yout_ini = yout.copy()

        if i%10 == 0:
            fig, axs = plt.subplots(4, 1, figsize=(20, 40))
            ax = axs[0]
            fcs.set_axes_format(ax, r'Time index', r'Displacement')
            ax.plot(yref, linewidth=1.0, linestyle='--', label=r'current')
            ax.plot(yref_ini, linewidth=1.0, linestyle='-', label=r'initial')
            ax.legend(fontsize=14)

            ax = axs[1]
            fcs.set_axes_format(ax, r'Time index', r'Displacement')
            ax.plot(yout, linewidth=1.0, linestyle='--', label=r'current')
            ax.plot(yout_ini, linewidth=1.0, linestyle='-', label=r'initial')
            ax.legend(fontsize=14)

            ax = axs[2]
            fcs.set_axes_format(ax, r'Time index', r'Input')
            ax.plot(u, linewidth=1.0, linestyle='-')

            ax = axs[3]
            fcs.set_axes_format(ax, r'Time index', r'disturbance')
            ax.plot(dy.flatten(), linewidth=1.0, linestyle='-')

            if is_save is True:
                plt.savefig(os.path.join(path_train_fig,str(i)+'.pdf'))
                plt.close()
            else:
                plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    # fcs.set_axes_format(ax, r'Index', r'Hidden state')
    # for i in range(len(s_list)):
    #     if i%50 == 0:
    #         s = s_list[i]
    #         ax.plot(s, linewidth=0.5, linestyle='-')
    # if is_save is True:
    #     plt.savefig(os.path.join(path_train_fig,'hidden_state.pdf'))
    #     plt.close()
    # else:
    #     plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    fcs.set_axes_format(ax, r'Iteration', r'Loss')
    ax.plot(loss_list, linewidth=1, linestyle='-')
    if is_save is True:
        plt.savefig(os.path.join(path_train_fig,'loss.pdf'))
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    test()