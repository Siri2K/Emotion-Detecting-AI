import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from typing import Union
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import DataLoader

from data import EmotionImages, ImageDataset
from src.CNN.CNNModel import CNNModel

from sklearn import metrics


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
    dataLoader["validation"] = DataLoader(dataset=ImageDataset(validation_data_list), batch_size=batch_size,
                                          shuffle=False)

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

    loss_list = []
    acc_list = []

    for epoch in range(numEpoch):
        for batch_idx, (images, labels) in enumerate(dataLoader["train"]):
            # Move data to device (GPU if available)
            images = images.to(device)
            labels = torch.tensor([emotion_to_index[label] for label in labels]).to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            # Print progress
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{numEpoch}], "
                    f"Batch [{batch_idx + 1}/{len(dataLoader['train'])}],"
                    f" Loss: {loss.item()}, "
                    f" Accuracy: {(correct / total) * 100}%"
                )

    train_accuracy(dataLoader, model)


def train_accuracy(dataLoader: dict, model: Union[CNNModel]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}
    index_to_emotion = {v: k for k, v in emotion_to_index.items()}
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        for images, labels in dataLoader['test']:
            images = images.to(device)
            labels = torch.tensor([emotion_to_index[label] for label in labels]).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        print("Test Accuracy of the model on the 300 test images : {} %".format((correct / total) * 100))

        # Saving model
        projectDirectory: str = os.path.dirname(os.path.abspath(__file__))
        if projectDirectory.endswith("src"):
            desiredDirectory = os.path.join(os.path.dirname(projectDirectory), "resources")
        else:
            desiredDirectory = os.path.join(projectDirectory, "resources")

        torch.save(model.state_dict(), os.path.join(desiredDirectory, "model_location.pth"))

        confusion(all_labels, all_preds)


def confusion(c, p):
    emotionLabels = ['Angry', 'Focused', 'Happy', 'Neutral']
    confusion_matrix = metrics.confusion_matrix(c, p)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=emotionLabels)

    cm_display.plot()
    plt.show()


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
