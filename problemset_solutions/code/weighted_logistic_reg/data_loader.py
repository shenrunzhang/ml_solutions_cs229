import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X_file, y_file):
        self.X = torch.tensor(self._load_data(X_file), dtype=torch.float)
        self.y = torch.tensor(self._load_data(y_file), dtype=torch.float)

    def _load_data(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            data = [list(map(float, line.strip().split())) for line in lines]
        return data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__== '__main__':
    # Create an instance of the dataset
    dataset = CustomDataset(r'code_data\PS1-data\q2\data\x.dat', r'code_data\PS1-data\q2\data\y.dat')

    # Create a data loader to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Example usage: iterate over the data loader
    for inputs, labels in data_loader:
        # Process your data here
        print('---------------')
        print(inputs, labels)
        pass

    print(len(dataset))
    print(dataset[0])
    print(dataset.y[:])
    print(len(dataset.X))
    print(dataset.X[5,:])

    z = torch.zeros(5,1)
    print(dataset.y[67])
    z[3] = dataset.y[67] + dataset.y[67] + 3
    print(dataset.X[5] @ torch.tensor([1, 1], dtype=torch.float))

    array = torch.tensor([1,2,3,4])
    print(array.unsqueeze(0))

    array = torch.tensor([[1],[2]])
    print(array.squeeze())
    print(torch.zeros(5))
    print(torch.cuda.is_available())