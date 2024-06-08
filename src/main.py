import sys

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Union
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import DataLoader

from data import EmotionImages, ImageDataset
from src.CNN.CNNModel import CNNModel


# Preprocess Dataset
def processDataset(imageDictionary: EmotionImages):
    # Update Dataset
    batch_size = 64
    dataset = imageDictionary.getDataset()

    dataLoader: dict = {
        "train": {},
        "test": {},
        "validation": {}
    }

    # Setup Dataset
    train_data_list = []
    for emotion in dataset.get('train'):
        for item in dataset['train'][emotion]:
            transformed_image = ToTensor()(item)
            train_data_list.append({'image': transformed_image, 'label': emotion})

    test_data_list = []
    for emotion in dataset.get('test'):
        for item in dataset['test'][emotion]:
            transformed_image = ToTensor()(item)
            test_data_list.append({'image': transformed_image, 'label': emotion})

    validation_data_list = []
    for emotion in dataset.get('validation'):
        for item in dataset['validation'][emotion]:
            transformed_image = ToTensor()(item)
            validation_data_list.append({'image': transformed_image, 'label': emotion})

    dataLoader["train"] = DataLoader(dataset=ImageDataset(train_data_list), batch_size=batch_size, shuffle=False)
    dataLoader["test"] = DataLoader(dataset=ImageDataset(test_data_list), batch_size=batch_size, shuffle=False)
    dataLoader["validation"] = DataLoader(dataset=ImageDataset(validation_data_list), batch_size=batch_size, shuffle=False)

    return dataLoader


def trainCNN(dataLoader: dict, model: Union[CNNModel]):
    # Set device to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Loss Function and Optimizer
    numEpoch = 10
    criterion = nn.CrossEntropyLoss()
    model = CNNModel().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}

    for epoch in range(numEpoch):
        for batch_idx, (images, labels) in enumerate(dataLoader["train"]):
            # Move data to device (GPU if available)
            images = images.to(device)
            labels = torch.tensor([emotion_to_index[label] for label in labels]).to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{numEpoch}], Batch [{batch_idx + 1}/{len(dataLoader['train'])}],"
                      f" Loss: {loss.item()}")

def main():
    # Initialize DataSet
    dataset: EmotionImages = EmotionImages()
    dataset.initialize()

    # Choose to Either Clean or Visualize Dataset
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clean":
            dataset.cleanImages()
            print("Dataset Cleaned")
            return
        elif sys.argv[1] == "--display":
            dataset.plotVisuals()
            return
        else:
            print("Invalid Command")
            print("Please Enter : python main.py or python main.py --clean")
            return

    # Initialize CNN
    model: CNNModel = CNNModel()
    trainCNN(processDataset(dataset), model)


if __name__ == '__main__':
    main()