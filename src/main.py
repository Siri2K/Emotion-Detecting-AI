import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF


from typing import Union, List
from sklearn import metrics
from data import EmotionImages, DataLoader, Image, ImageDataset, ToTensor
from src.CNN.CNNModel import CNNModel
from src.CNN.CNNVariant1 import CNNVariant1
from src.CNN.CNNVariant2 import CNNVariant2
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import KFold


#kfold cross validation for CNN model 10 fold cross validation
def kfold_cross_validation(dataLoader: DataLoader, model: Union[CNNModel, CNNVariant1, CNNVariant2], device: torch.device,
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

    # Setup K fold cross validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Define the model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNVariant2().to(device)

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start the k-fold cross-validation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataLoader.dataset)):
        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(dataLoader.dataset, batch_size=10, sampler=train_subsampler)
        testloader = DataLoader(dataLoader.dataset, batch_size=10, sampler=test_subsampler)

        # Init the neural network
        model.apply(reset_weights)  # Function to reset weights if needed

        # Run the training loop for defined number of epochs
        for epoch in range(0, 10):
            # Train the model
            model.train()
            for i, data in enumerate(trainloader):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

            # Evaluation for this fold
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            print(f'Fold {fold}, Epoch {epoch}, Accuracy: {100.0 * correct / total}%')

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


# select a random image
#help of chatgpt and lab 6
def random_image(data_loader: DataLoader, models: List[Union[CNNModel, CNNVariant1, CNNVariant2]]):
    # Select a random image from the dataset
    dataset: ImageDataset = data_loader.dataset
    emotionLabels = dataset.getAllLabels()

    for model in models:
        idx = np.random.randint(0, dataset.getDataSize())  # Get a random index
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
            break
        elif accuracy < prevAccuracy and numberOfFails > 2:
            break
        elif accuracy < prevAccuracy:
            numberOfFails += 1
        else:
            prevAccuracy = accuracy

        # Test Model for validation [2]
        all_labels, all_preds = testCNN(dataLoader=dataLoader[2], model=model, device=device, savePath=savePath)

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
            print(f"Current Test Accuracy : {savedData['accuracy']:.4f}")
    elif isinstance(model, CNNVariant1):
        if os.path.exists(savePath):
            print(f"Current Test Accuracy : {savedData['accuracy']:.4f}")
    elif isinstance(model, CNNVariant2):
        if os.path.exists(savePath):
            print(f"Current Test Accuracy : {savedData['accuracy']:.4f}")

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


def main():
    # Initialize DataSet
    dataset = EmotionImages()
    dataset.setup()

    # Choose to Either Clean or Visualize Dataset
    models = []
    if len(sys.argv) > 1:
        if sys.argv[1] == "--display":
            dataset.plotVisuals()
            return
        elif sys.argv[1] == "--variant1":
            models.append(CNNVariant1())
        elif sys.argv[1] == "--variant2":
            models.append(CNNVariant2())
        elif sys.argv[1] == "--trainAll":
            models.append(CNNModel())
            models.append(CNNVariant1())
            models.append(CNNVariant2())
        elif sys.argv[1] == "--none":
            pass
        else:
            print("Invalid Command")
            print("Please Enter : python main.py or python main.py --display")
            return
    else:
        models.append(CNNModel())

    # Gather Inputs
    train_dataloader, test_dataloader, validation_dataloader = dataset.getDataLoaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveFile: str = ''

    kfold_cross_validation(dataLoader=train_dataloader, model=CNNVariant2(), device=device,
                           savePath=os.path.join(dataset.getDataDirectory(), "bin", "variant2.pth"))

    # Train All Models
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

    if len(models) > 0:
        random_image(test_dataloader, models)
    else:
        random_image(test_dataloader, [CNNModel()])

    # Display Dataset
    plt.show()


if __name__ == '__main__':
    main()
