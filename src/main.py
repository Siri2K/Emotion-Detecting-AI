import os
import sys
from typing import Union, List

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


def trainCNN(dataLoader: List[DataLoader], model: Union[CNNModel, CNNVariant1, CNNVariant2], device: torch.device,
             savePath: str):

    # Determine Model to Train
    modelInst: str = ""
    savedData = torch.load(savePath) if os.path.exists(savePath) else None
    if isinstance(model, CNNModel):
        modelInst: str = "Model"
        if os.path.exists(savePath):
            model.load_state_dict(savedData['model_state'])
    elif isinstance(model, CNNVariant1):
        modelInst: str = "Model"
        if os.path.exists(savePath):
            model.load_state_dict(savedData['model_state'])
    elif isinstance(model, CNNVariant2):
        modelInst: str = "Model"
        if os.path.exists(savePath):
            model.load_state_dict(savedData['model_state'])

    numEpoch = 10


    # Setup Loss Factor and Optimiser
    model = model.to(device=device)  # Configure GPU and Train Mode
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Setup Predicted
    prevAccuracy = 0
    accuracy = 0
    all_labels = []
    all_preds = []
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}

    # Setup Labels
    for epoch in range(numEpoch):
        model.train()
        for batch_idx, (images, labels) in enumerate(dataLoader[0]):
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
                    f" Loss: {loss.item():.4f}, "
                    f" Accuracy: {accuracy:.4f}%"
                )

        # Update Accuracy
        if accuracy == 100 or accuracy < prevAccuracy:
            break
        else:
            prevAccuracy = accuracy

        all_labels, all_preds = testCNN(dataLoader=dataLoader[1], model=model, device=device, savePath=savePath)

    # Kill Code
    displayPerformanceMetrics(all_labels, all_preds)
    confusion(all_labels, all_preds, modelInst)


def testCNN(dataLoader: DataLoader, model: Union[CNNModel, CNNVariant1, CNNVariant2], device: torch.device,
            savePath: str):
    # Configure CNN Device and Mode
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}

    # Determine Model To Test
    savedData = torch.load(savePath) if os.path.exists(savePath) else None
    if isinstance(model, CNNModel):
        if os.path.exists(savePath):
            model = CNNModel()
            model.load_state_dict(savedData['model_state'])
            print(f"Current Test Accuracy : {savedData['accuracy']}")
    elif isinstance(model, CNNVariant1):
        if os.path.exists(savePath):
            model = CNNModel()
            model.load_state_dict(savedData['model_state'])
            print(f"Current Test Accuracy : {savedData['accuracy']}")
    elif isinstance(model, CNNVariant2):
        if os.path.exists(savePath):
            model = CNNModel()
            model.load_state_dict(savedData['model_state'])
            print(f"Current Test Accuracy : {savedData['accuracy']}")

    model = model.to(device=device).eval()

    # Evaluate Dataset
    with torch.no_grad():
        # Setup Prediction Count
        correct: int = 0
        total: int = 0

        # Initialize labels
        all_labels = []
        all_preds = []

        # Gather Evaluation Data
        bestAccuracy: float = savedData['accuracy'] if os.path.exists(savePath) else 0
        accuracy: float = 0
        for images, labels in dataLoader:
            images = images.to(device)
            labels = torch.tensor([emotion_to_index[label] for label in labels]).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Evaluate Obtained Data
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Calculate Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = (correct / total) * 100

        if accuracy == 100 or accuracy > bestAccuracy:
            print(f"New Test Accuracy : {accuracy:.4f}%")
            modelData: dict = {
                'model_state': model.state_dict(),
                'accuracy': accuracy
            }
            torch.save(modelData, savePath)

    return all_labels, all_preds


def displayPerformanceMetrics(actual, predicted):
    # Performance Metrics
    recall_macro = recall_score(actual, predicted, average='macro')
    print(f"Recall_Macro : {(recall_macro * 100):.4f}%")

    precision_macro = precision_score(actual, predicted, average='macro')
    print(f"Precision_Macro : {(precision_macro * 100):.4f}%")

    f1_macro = f1_score(actual, predicted, average='macro')
    print(f"F1_Macro : {(f1_macro * 100):.4f}%")

    recall_micro = recall_score(actual, predicted, average='micro')
    print(f"Recall_Micro : {(recall_micro * 100):.4f}%")

    precision_micro = precision_score(actual, predicted, average='micro')
    print(f"Precision_Micro : {(precision_micro * 100):.4f}%")

    f1_micro = f1_score(actual, predicted, average='micro')
    print(f"F1_Micro : {(f1_micro * 100):.4f}% \n\n")


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
        trainCNN(dataLoader=[train_dataloader, test_dataloader, validation_dataloader], model=model, device=device,
                 savePath=os.path.join(dataset.getDataDirectory(), "bin", saveFile))

    plt.show()


if __name__ == '__main__':
    main()
