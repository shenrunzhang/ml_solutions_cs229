import torch
from torch import zeros
from matplotlib import pyplot as plt
from data_loader import CustomDataset
import numpy
import time


def lwlr(X: torch.Tensor, y: torch.Tensor, x: torch.Tensor, tau) -> int:
    lam = 0.0001
    theta = torch.tensor([0, 0], device=X.device, dtype=torch.float)

    # number of examples
    m = len(X)

    # define weights
    w = []

    for i in range(m):
        w.append(torch.exp(- (torch.norm(X[i] - x) ** 2) / (2 * tau ** 2)))

    zed = torch.zeros(m, 1, device=X.device)

    def z(theta: torch.Tensor) -> torch.Tensor:
        for i in range(m):
            zed[i] = w[i] * (y[i] - torch.sigmoid(theta @ X[i]))

        return zed

    def grad(theta: torch.Tensor) -> torch.Tensor:
        return X.T @ z(theta) - lam * theta.unsqueeze(1)

    def hessian(theta: torch.Tensor) -> torch.Tensor:
        return X.T @ D(theta) @ X - lam * torch.eye(2, device=X.device)

    def D(theta: torch.Tensor) -> torch.Tensor:
        d = torch.zeros(m, m, device=X.device)
        for i in range(m):
            d[i, i] = - w[i] * torch.sigmoid(theta @ X[i]) * (1 - torch.sigmoid(theta @ X[i]))
        return d

    # update theta for 100 steps
    for _ in range(20):
        theta -= torch.linalg.solve(hessian(theta), grad(theta)).squeeze()

    # indicator 1{h(x) > 0.5}
    return 1 if torch.sigmoid(theta @ x) > 0.5 else 0


def plot_lwlr(X: torch.Tensor, y: torch.Tensor, tau: float, res: int, device) -> None:
    x = torch.zeros(2, device=device)
    pred = torch.zeros(res, res, device=device)

    for i in range(res):
        start_time = time.time()
        for j in range(res):
            x[0] = 2 * i / (res - 1) - 1
            x[1] = 2 * j / (res - 1) - 1
            pred[j, i] = lwlr(X, y, x, tau)

        print("time taken for 10 runs: ", time.time() - start_time)

    pred = torch.round(pred).to(torch.int32)
    print(pred)
    plt.figure(1)

    # Create a colormap with blue for 0 and orange for 1
    cmap = plt.cm.colors.ListedColormap(['blue', 'orange'])

    # Plot the matrix using imshow
    plt.imshow(pred, cmap=cmap, interpolation='nearest')

    plt.scatter((res / 2) * (1 + X[y.flatten() == 0, 0]) + 0.5, (res / 2) * (1 + X[y.flatten() == 0, 1]) + 0.5,
                color='black', marker='o')
    plt.scatter((res / 2) * (1 + X[y.flatten() == 1, 0]) + 0.5, (res / 2) * (1 + X[y.flatten() == 1, 1]) + 0.5,
                color='black', marker='x')
    plt.text(res / 2 - res / 7, - res / 15, 'tau = {}'.format(tau), fontsize=18)
    plt.show()


if __name__ == '__main__':
    # Create an instance of the dataset
    dataset = CustomDataset(r'code_data\PS1-data\q2\data\x.dat', r'code_data\PS1-data\q2\data\y.dat')

    # Create a data loader to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
        dataset.X = dataset.X.to(device)
        dataset.y = dataset.y.to(device)
    else:
        device = torch.device('cpu')
        print('Using CPU')

    tau = 0.5

    plot_lwlr(dataset.X, dataset.y, tau, 50, device)
