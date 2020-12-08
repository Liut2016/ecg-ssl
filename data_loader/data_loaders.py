from torchvision import datasets, transforms
from base import BaseDataLoader
import torch.utils.data as Data
import scipy.io
import torch

# MNIST数据集
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# MIT-BIH数据集
class MitbihDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        data = scipy.io.loadmat(self.data_dir)
        if training:
            x = torch.from_numpy(data['X_train'])
            y = torch.from_numpy(data['Y_train']).permute(1, 0)
        else:
            x = torch.from_numpy(data['X_test'])
            y = torch.from_numpy(data['Y_test']).permute(1, 0)

        x = x.float()
        x = x.permute(0, 2, 1)
        # x = x.reshape(-1, 1, x.shape[1], x.shape[2])
        y = y.squeeze()
        y = y.long()

        self.dataset = Data.TensorDataset(x, y)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)