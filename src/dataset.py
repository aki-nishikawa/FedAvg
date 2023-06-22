r""" dataset
"""

from typing import Dict, List

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


class FedAvgRetriever():
    r"""
    """

    def __init__(self, config: Dict) -> None:

        self.config = config

        # init CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)

    def get(self, num_clients: int) -> List[Dict[str, DataLoader]]:
        r"""
        """
        train_size = len(self.train_set) // num_clients
        test_size = len(self.test_set) // num_clients

        dataloaders = []
        for i in range(num_clients):
            train_idx = range(i * train_size, (i + 1) * train_size)
            test_idx = range(i * test_size, (i + 1) * test_size)

            train_set = Subset(self.train_set, train_idx)
            test_set = Subset(self.test_set, test_idx)

            train_loader = DataLoader(
                train_set, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)
            test_loader = DataLoader(
                test_set, batch_size=self.config["batch_size"], shuffle=False, num_workers=4)
            dataloaders.append({
                "train": train_loader,
                "test": test_loader
            })

        return dataloaders
