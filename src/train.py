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

        # Chatgpt recomended us to check if Dataloader is empty
        if len(trainLoader) == 0:
            print("Training DataLoader is empty.")
            return

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

        print(f"Starting training with {len(trainLoader)} batches")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for i, (images, labels) in enumerate(trainLoader):
                # Check the shape of images tensor
                if images.dim() == 3:
                    # Add batch dimension if missing
                    images = images.unsqueeze(0)

                # Ensure images have 4 dimensions before applying permute
                if images.dim() == 4:
                    images = images.permute(0, 3, 1, 2)  # Convert to channel-first format

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)
                if (i + 1) % 100 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {correct / total * 100:.2f}%')