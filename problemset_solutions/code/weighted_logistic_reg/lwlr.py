import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_loader import CustomDataset
from math import exp
from torch.nn import Sigmoid
from torch import zeros, sigmoid, tensor, eye
from torch.linalg import *
import time

def lwlr(X: torch.Tensor, y:torch.Tensor, x: torch.Tensor, tau) -> int:
    '''
    Input:
    X - features
    y - labels
    x - query point
    tau - bandwidth parameter

    Output:
    classifies point x as 0 or 1
    '''
    lam = 0.0001
    theta = tensor([0,0], dtype=torch.float)

    # number of examples
    m = len(X)

    # define weights
    w = []
    
    for i in range(m):
        w.append(exp(- (norm(X[i] - x) ** 2 ) / (2 * tau ** 2)))

    zed = zeros(m,1)
    d = zeros(m,m)

    def z(theta: torch.Tensor) -> torch.Tensor:
        for i in range(m):
            zed[i] = w[i] * (y[i] - sigmoid(theta @ X[i]))
        return zed

    def grad(theta: torch.Tensor) -> torch.Tensor:
        grad =  X.T @ z(theta) - lam * theta.unsqueeze(1)
        return grad
    
    def hessian(theta: torch.Tensor) -> torch.Tensor:
        hess =  (X.T @ D(theta)) @ X - lam * eye(2)
        return hess
    
    def D(theta:torch.Tensor) -> torch.Tensor:

        for i in range(m):
            d[i,i] = - w[i] * sigmoid(theta @ X[i]) * (1 - sigmoid(theta @ X[i]))
        
        return d
    
    # update theta for 100 steps
    for _ in range(2):
        theta -= solve(hessian(theta), grad(theta)).squeeze()

    # indicator 1{h(x) > 0.5}
    return 1 if sigmoid(theta @ x) > 0.5 else 0

if __name__ == '__main__':
        
    # Create an instance of the dataset
    dataset = CustomDataset(r'code_data\PS1-data\q2\data\x.dat', r'code_data\PS1-data\q2\data\y.dat')

    # Create a data loader to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    tau = 0.5
    start_time = time.time()
    num = lwlr(dataset.X, dataset.y, tensor([-0.5, 0]), 0.5)
    print((time.time() - start_time) * 50)
