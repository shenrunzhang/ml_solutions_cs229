import torch
from torch import zeros
from matplotlib import pyplot as plt
from lwlr import lwlr
from data_loader import CustomDataset
import numpy
import time


def plot_lwlr(X: torch.Tensor, y: torch.Tensor, tau: float, res:int) -> None:
    x = zeros(2)
    pred = zeros(res, res)

    for i in range(res):
        for j in range(res):
            x[0] = 2*(i)/(res-1) - 1
            x[1] = 2*(j)/(res-1) - 1 
            pred[j, i] = lwlr(X, y, x, tau)
        print(i)

    pred = torch.round(pred).to(torch.int32)   

    plt.figure(1)

    # Create a colormap with blue for 0 and orange for 1
    cmap = plt.cm.colors.ListedColormap(['azure', 'orange'])

    # Plot the matrix using imshow
    plt.imshow(pred, cmap=cmap, interpolation='nearest', origin='lower')
    plt.scatter((res/2) * (1 + X[y.flatten() == 0, 0]) + 0.5, (res/2) * (1 + X[y.flatten() == 0, 1]) + 0.5, color='black', marker='o')
    plt.scatter((res/2) * (1 + X[y.flatten() == 1, 0]) + 0.5, (res/2) * (1 + X[y.flatten() == 1, 1]) + 0.5, color='black', marker='x')
    plt.text(res/2 - res/7, 1.05 * res, 'tau = {}'.format(tau), fontsize=18)
    plt.savefig('res_%d_tau_%s.png'%(res,'point'.join(str(tau).split('.'))))
    plt.close(1)


if __name__ == '__main__':
    # Create an instance of the dataset
    dataset = CustomDataset(r'code_data\PS1-data\q2\data\x.dat', r'code_data\PS1-data\q2\data\y.dat')

    # Create a data loader to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    taus = [0.05, 0.5, 1, 5]

    for tau in taus:
        plot_lwlr(dataset.X, dataset.y, tau, 100)
