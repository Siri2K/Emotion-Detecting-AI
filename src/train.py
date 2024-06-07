import torch

import torch.utils.data as td
import torch.nn as nn

from typing import Union
from data import EmotionImages
from CNN.CNNModel import CNNModel
from CNN.CNNVariant1 import CNNVariant1
from CNN.CNNVariant2 import CNNVariant2


class Train:
    def __init__(self, dataset: EmotionImages, model: Union[CNNModel, CNNVariant1, CNNVariant2]):
        self.__model: Union[CNNModel, CNNVariant1, CNNVariant2] = model
        self.__trainingDataSet: td.DataLoader = dataset.getDataLoader()['train']
        self.__testingDataSet: td.DataLoader = dataset.getDataLoader()['test']
        self.__validationDataSet: td.DataLoader = dataset.getDataLoader()['validation']

    def getModel(self) -> Union[CNNModel, CNNVariant1, CNNVariant2]:
        return self.__model

    def getTrainingDataSet(self) -> td.DataLoader:
        return self.__trainingDataSet

    def getTestingDataSet(self) -> td.DataLoader:
        return self.__testingDataSet

    def getValidationDataSet(self) -> td.DataLoader:
        return self.__validationDataSet

    def train(self):
        # Data
        trainLoader = self.getTrainingDataSet()
        model = self.getModel()

        # Hidden Parameters
        num_epochs = 10
        batch_size = 4
        num_classes = 4
        learning_rate = 0.001

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(trainLoader)
        loss_list = []
        acc_list = []

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(trainLoader):  # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # Backprop and optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Train accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100)
                    )
