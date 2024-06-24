import os
from typing import List, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import KFold
from torch import nn, optim
from emotionData import DataLoader, ImageDataset, ToTensor
from src.CNN.CNNModel import CNNModel
from src.CNN.CNNVariant1 import CNNVariant1
from src.CNN.CNNVariant2 import CNNVariant2


# select a random image
#help of chatgpt and lab 6
def random_image(data_loader: DataLoader, models: List[Union[CNNModel, CNNVariant1, CNNVariant2]]):
    """

    :param data_loader:
    :param models:
    :return:
    """

    # Select a random image from the dataset
    dataset: ImageDataset = data_loader.dataset
    emotionLabels = dataset.getAllLabels()

    for model in models:
        idx = np.random.randint(0, dataset.getDataSize()-1)  # Get a random index
        img, label = dataset.getData(idx)  # Select the image

        # Convert tensor to PIL Image for displaying
        img_pil = TF.to_pil_image(img)

        # Convert back to tensor and normalize/prepare for model
        img_tensor = ToTensor()(img_pil)

        # Add batch dimension (model expects batch, not single image)
        img_tensor = img_tensor.unsqueeze(0)

        # Predict with model
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)

        # Display the image
        plt.figure()
        plt.title(f" Actual: {label} |" +
                  f" Predicted: {emotionLabels[predicted.item()]}")
        plt.imshow(img_pil, cmap='gray')  # Displaying the PIL image
        plt.axis('off')  # Turn off axis numbers and ticks


# Train CNN Model
def trainCNN(dataLoader: List[DataLoader], model: Union[CNNModel, CNNVariant1, CNNVariant2], device: torch.device,
             savePath: str):
    # Determine Model to Train
    modelInst: str = ""
    savedData = torch.load(savePath) if os.path.exists(savePath) else None
    if isinstance(model, CNNModel):
        modelInst: str = "Model"
    elif isinstance(model, CNNVariant1):
        modelInst: str = "Variant1"
    elif isinstance(model, CNNVariant2):
        modelInst: str = "Variant2"

    # Setup Loss Factor and Optimiser
    numberOfFails = 0
    numEpoch = 15
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
    print(modelInst)
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
        if accuracy == 100:
            print()
            break
        elif accuracy < prevAccuracy and numberOfFails > 2:
            print()
            break
        elif accuracy < prevAccuracy:
            numberOfFails += 1
        else:
            prevAccuracy = accuracy

        # Test Model for validation [2]
        all_labels, all_preds = testCNN(dataLoader=dataLoader[2], model=model, device=device, savePath=savePath,
                                        saveModel=False)

    # Kill Code
    displayPerformanceMetrics(all_labels, all_preds)
    confusion(all_labels, all_preds, modelInst)


def trainCNNWithKFold(dataLoader: List[DataLoader], model: Union[CNNModel, CNNVariant1, CNNVariant2],
                      device: torch.device, savePath: str):

    """
    For K-Fold implementation, ChatGPT was used

    :param dataLoader:
    :param model:
    :param device:
    :param savePath:
    :return:
    """

    # Determine Model to Train
    modelInst: str = ""
    savedData = torch.load(savePath) if os.path.exists(savePath) else None
    if isinstance(model, CNNModel):
        modelInst: str = "Model"
    elif isinstance(model, CNNVariant1):
        modelInst: str = "Variant1"
    elif isinstance(model, CNNVariant2):
        modelInst: str = "Variant2"

    # Setup Loss Factor and Optimiser:
    numEpoch = 15
    emotions = ['Angry', 'Happy', 'Focused', 'Neutral']
    emotionToIndex = {emotion: index for index, emotion in enumerate(emotions)}
    criterion = nn.CrossEntropyLoss()

    # Setup KFolds
    kFold = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation
    all_labels = []
    all_preds = []

    for fold_idx, (train_idx,test_idx) in enumerate(kFold.split(dataLoader[0].dataset.data)):
        print(f"\nFold {fold_idx + 1}:")

        # Setup Model for each fold
        model = model.to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_loader = DataLoader(dataLoader[0].dataset, batch_size=dataLoader[0].batch_size, shuffle=False,
                                  sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        test_loader = DataLoader(dataLoader[0].dataset, batch_size=dataLoader[0].batch_size, shuffle=False,
                                 sampler=torch.utils.data.SubsetRandomSampler(test_idx))

        # Gather Accuracy
        prevAccuracy = 0
        accuracy = 0
        numberOfFails = 0

        # Train the CNN Model
        for epoch in range(numEpoch):
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Move data to device (GPU if available)
                images = images.to(device)
                labels = torch.tensor([emotionToIndex[label] for label in labels]).to(device)

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
            if accuracy == 100:
                break
            elif accuracy < prevAccuracy and numberOfFails > 2:
                break
            elif accuracy < prevAccuracy:
                numberOfFails += 1
            else:
                prevAccuracy = accuracy

            # Perform evaluation on the test set for each fold
            all_labels, all_preds = testCNN(dataLoader=test_loader, model=model, device=device, savePath=savePath,
                                            saveModel=False)

            # Kill Code
            print(f"Fold {fold_idx + 1} Performance:")
            displayPerformanceMetrics(all_labels, all_preds)
        confusion(all_labels, all_preds, modelInst)


def testCNN(dataLoader: DataLoader, model: Union[CNNModel, CNNVariant1, CNNVariant2], device: torch.device,
            savePath: str, saveModel:bool = True):
    # Configure CNN Device and Mode
    emotion_to_index = {'Angry': 0, 'Happy': 1, 'Focused': 2, 'Neutral': 3}

    # Determine Model To Test
    savedData = torch.load(savePath) if os.path.exists(savePath) else None
    if isinstance(model, CNNModel):
        if os.path.exists(savePath):
            print(f"Saved Test Accuracy : {savedData['accuracy']:.4f}")
    elif isinstance(model, CNNVariant1):
        if os.path.exists(savePath):
            print(f"Saved Test Accuracy : {savedData['accuracy']:.4f}")
    elif isinstance(model, CNNVariant2):
        if os.path.exists(savePath):
            print(f"Saved Test Accuracy : {savedData['accuracy']:.4f}")

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

        if (accuracy == 100 or accuracy > bestAccuracy) and saveModel:
            print(f"New Test Accuracy : {accuracy:.4f}%")
            modelData: dict = {
                'model_state': model.state_dict(),
                'accuracy': accuracy
            }
            torch.save(modelData, savePath)
        else:
            print(f"Epoch Test Accuracy : {accuracy:.4f}%")

    return all_labels, all_preds


def displayPerformanceMetrics(actual, predicted):
    # Performance Metrics
    recall_macro = recall_score(actual, predicted, average='macro',zero_division=1)
    print(f"Recall_Macro : {(recall_macro * 100):.4f}%")

    precision_macro = precision_score(actual, predicted, average='macro',zero_division=1)
    print(f"Precision_Macro : {(precision_macro * 100):.4f}%")

    f1_macro = f1_score(actual, predicted, average='macro',zero_division=1)
    print(f"F1_Macro : {(f1_macro * 100):.4f}%")

    recall_micro = recall_score(actual, predicted, average='micro',zero_division=1)
    print(f"Recall_Micro : {(recall_micro * 100):.4f}%")

    precision_micro = precision_score(actual, predicted, average='micro',zero_division=1)
    print(f"Precision_Micro : {(precision_micro * 100):.4f}%")

    f1_micro = f1_score(actual, predicted, average='micro',zero_division=1)
    print(f"F1_Micro : {(f1_micro * 100):.4f}% \n")


def confusion(actual, predicted, modelName: str):
    """
    # w3 school  https://www.w3schools.com/python/python_ml_confusion_matrix.asp

    :param modelName:
    :param actual:models
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