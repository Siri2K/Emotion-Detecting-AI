import os
import sys
from typing import Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

from data import EmotionImages, ImageDataset
from src.CNN.CNNModel import CNNModel

# To calculate time
import time


def trainCNN(dataLoader: EmotionImages, model: Union[CNNModel]):
    start = time.time()

    # Set device to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, _, _ = dataLoader.getDataLoaders()

    # Setup Loss Function and Optimizer
    numEpoch = 10
    criterion = nn.CrossEntropyLoss()
    model = model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}

    for epoch in range(numEpoch):
        model.train()  # Ensure the model is in training mode
        for batch_idx, (images, labels) in enumerate(train_dataloader):
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
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                print(
                    f"Epoch [{epoch + 1}/{numEpoch}], "
                    f"Batch [{batch_idx + 1}/{len(train_dataloader)}],"
                    f" Loss: {loss.item()}, "
                    f" Accuracy: {(correct / total) * 100}%"
                )

    print(f"Training CNN Time: {time.time() - start}")
    train_accuracy(dataLoader, model)


def train_accuracy(dataLoader: EmotionImages, model: Union[CNNModel]):
    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_dataloader, _ = dataLoader.getDataLoaders()
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = torch.tensor([emotion_to_index[label] for label in labels]).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        print("Test Accuracy of the model on the test images : {} %".format((correct / total) * 100))

        model_path = os.path.join(dataLoader.getDataDirectory(), "model_location.pth")
        torch.save(model.state_dict(), model_path)

        print(f"Training Accuracy Time: {time.time() - start}")
        confusion(all_labels, all_preds)

#w3 school  https://www.w3schools.com/python/python_ml_confusion_matrix.asp
def confusion(actual, predicted):
    start = time.time()
    emotion_labels = ['Angry', 'Happy', 'Focused', 'Neutral']
    print(f"Actual labels: {actual}")
    print(f"Predicted labels: {predicted}")
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=emotion_labels)
    print(f"Confusion Matrix Plotting Time: {time.time() - start}")
    cm_display.plot()
    plt.show()



def main():
    start = time.time()

    # Initialize DataSet
    dataset = EmotionImages()
    dataset.setup()

    # Choose to Either Clean or Visualize Dataset
    if len(sys.argv) > 1:
        if sys.argv[1] == "--display":
            dataset.plotVisuals()
            return
        else:
            print("Invalid Command")
            print("Please Enter : python main.py or python main.py --clean")
            return

    # Initialize CNN
    print(f"Dataset Setup Time: {time.time() - start}")
    model = CNNModel()
    trainCNN(dataset, model)


if __name__ == '__main__':
    main()
