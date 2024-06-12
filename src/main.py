import os
import sys
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

from data import EmotionImages, DataLoader
from src.CNN.CNNModel import CNNModel
from src.CNN.CNNVariant1 import CNNVariant1
from src.CNN.CNNVariant2 import CNNVariant2

from sklearn.metrics import recall_score, precision_score, f1_score


def trainCNN(dataLoader: DataLoader, model: Union[CNNModel, CNNVariant1, CNNVariant2], device: torch.device,
             savePath: str):
    # Determine Model To Train
    modelInst = ""
    if isinstance(model, CNNModel):
        modelInst: str = "Model"
    elif isinstance(model, CNNVariant1):
        modelInst: str = "Variant1"
    elif isinstance(model, CNNVariant2):
        modelInst: str = "Variant2"

    # Setup Training Inputs
    numEpoch = 10
    model = model.to(device=device)  # Configure GPU and Train Mode

    # Load Saved Model if exists
    if os.path.exists(savePath):
        print(f"Using Saved {modelInst}")
        model.load_state_dict(torch.load(savePath))
    else:
        print(f"Creating New {modelInst}")

    # Setup Loss Factor and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Setup Predicted
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}

    # Setup Labels
    prevAccuracy = 0
    accuracy: float = 0
    for epoch in range(numEpoch):
        model.train()
        for batch_idx, (images, labels) in enumerate(dataLoader):
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
                accuracy = (correct / total) * 100
                print(
                    f"Epoch [{epoch + 1}/{numEpoch}], "
                    f" Loss: {loss.item()}, "
                    f" Accuracy: {accuracy}%"
                )
        if accuracy < prevAccuracy:
            break
        else:
            prevAccuracy = accuracy


def testCNN(dataLoader: DataLoader, model: Union[CNNModel, CNNVariant1, CNNVariant2], device: torch.device,
            savePath: str):
    # Determine Model To Test
    modelInst = ""
    if isinstance(model, CNNModel):
        modelInst: str = "Model"
    elif isinstance(model, CNNVariant1):
        modelInst: str = "Variant1"
    elif isinstance(model, CNNVariant2):
        modelInst: str = "Variant2"

    # Configure CNN Device and Mode
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}
    model = model.to(device=device).eval()  # Configure GPU and Train Mode

    # Evaluate Dataset
    with torch.no_grad():
        # Setup Prediction Count
        correct: int = 0
        total: int = 0

        # Initialize labels
        all_labels = []
        all_preds = []

        # Gather Evaluation Data
        for images, labels in dataLoader:
            images = images.to(device)
            labels = torch.tensor([emotion_to_index[label] for label in labels]).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Evaluate Obtained Data
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        # Test Accuracy
        print(f"Test Accuracy : {(correct / total) * 100}% \n")

        # Performance Metrics
        recall_macro = recall_score(all_labels, all_preds, average='macro')
        print(f"Recall_Macro : {recall_macro * 100}%")

        precision_macro = precision_score(all_labels, all_preds, average='macro')
        print(f"Precision_Macro : {precision_macro * 100}%")

        f1_macro = f1_score(all_labels, all_preds, average='macro')
        print(f"F1_Macro : {f1_macro * 100}% \n")

        recall_micro = recall_score(all_labels, all_preds, average='micro')
        print(f"Recall_Micro : {recall_micro * 100}%")

        precision_micro = precision_score(all_labels, all_preds, average='micro')
        print(f"Precision_Micro : {precision_micro * 100}%")

        f1_micro = f1_score(all_labels, all_preds, average='micro')
        print(f"F1_Micro : {f1_micro * 100}% \n")

        # Save Model and Generate Confusion Matrix
        torch.save(model.state_dict(), savePath)
        confusion(all_labels, all_preds, modelInst)


def confusion(actual, predicted, modelName: str):
    """
    # w3 school  https://www.w3schools.com/python/python_ml_confusion_matrix.asp

    :param modelName:
    :param actual:
    :param predicted:
    :return:
    """

    # Create Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=['Angry', 'Happy', 'Focused', 'Neutral'])

    cm_display.plot()
    plt.title(f"Confusion Matrix for {modelName}")


def main():
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
            print("Please Enter : python main.py or python main.py --display")
            return

    # Gather Inputs
    train_dataloader, test_dataloader, validation_dataloader = dataset.getDataLoaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveFile: str = ''

    # Train All Models
    models = [CNNModel(), CNNVariant1(), CNNVariant2()]

    for model in models:
        if isinstance(model, CNNModel):
            saveFile: str = "model.pth"
        elif isinstance(model, CNNVariant1):
            saveFile: str = "variant1.pth"
        elif isinstance(model, CNNVariant2):
            saveFile: str = "variant2.pth"

        # Train & Test CNN Model
        trainCNN(dataLoader=train_dataloader, model=model, device=device,
                 savePath=os.path.join(dataset.getDataDirectory(), "bin", saveFile))
        testCNN(dataLoader=test_dataloader, model=model, device=device,
                savePath=os.path.join(dataset.getDataDirectory(), "bin", saveFile))

    plt.show()


if __name__ == '__main__':
    main()
