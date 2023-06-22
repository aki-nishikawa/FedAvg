from typing import Dict, List

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from src.models import CNNModel


class FedAvgClient():
    r""" FedAvgClient
    """

    def __init__(self, config: Dict, dataloaders: Dict[str, DataLoader]) -> None:
        self.config = config
        self.train_loader = dataloaders["train"]
        self.test_loader = dataloaders["test"]

        self.model: CNNModel = None  # type: ignore

    def set_global_model(self, global_model: CNNModel):
        del self.model
        self.model = global_model

    def train_local_model(self) -> CNNModel:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.config["learning_rate"])

        def _train_loop():
            running_loss = 0
            for i, (X, y) in enumerate(self.train_loader):
                X = X.to(self.config["device"])
                y = y.to(self.config["device"])
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            return running_loss / len(self.train_loader)

        def _test_loop():
            labels = []
            preds = []
            running_loss = 0
            with torch.no_grad():
                for i, (X, y) in enumerate(self.test_loader):
                    X = X.to(self.config["device"])
                    y = y.to(self.config["device"])
                    output = self.model(X)
                    loss = criterion(output, y)
                    running_loss += loss.item()
                    labels.append(y.detach().cpu())
                    preds.append(output.detach().cpu())
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0).argmax(1)
            acc = torch.sum(preds == labels) / len(labels)
            return running_loss / len(self.test_loader), acc

        self.model.to(self.config["device"])
        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = _train_loop()
            self.model.eval()
            test_loss, acc = _test_loop()

        return self.model


class FedAvgServer():
    r""" FedAvgServer
    """

    def __init__(self, config: Dict) -> None:
        self.config = config

    def init_global_model(self) -> CNNModel:
        global_model = CNNModel()
        return global_model

    def aggregate(self, local_models: List[CNNModel]) -> CNNModel:
        global_model = CNNModel().to(self.config["device"])

        global_model.conv1.weight.data = torch.zeros_like(
            global_model.conv1.weight.data)
        for local_model in local_models:
            global_model.conv1.weight.data += local_model.conv1.weight.data

        global_model.conv2.weight.data = torch.zeros_like(
            global_model.conv2.weight.data)
        for local_model in local_models:
            global_model.conv2.weight.data += local_model.conv2.weight.data

        global_model.fc1.weight.data = torch.zeros_like(
            global_model.fc1.weight.data)
        for local_model in local_models:
            global_model.fc1.weight.data += local_model.fc1.weight.data

        global_model.fc2.weight.data = torch.zeros_like(
            global_model.fc2.weight.data)
        for local_model in local_models:
            global_model.fc2.weight.data += local_model.fc2.weight.data

        return global_model
