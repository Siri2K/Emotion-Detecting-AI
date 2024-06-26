import os
import sys

from train import CNNModel, CNNVariant1, CNNVariant2
from train import plt
from train import torch
from train import trainCNN, trainCNNWithKFold, random_image
from emotionData import EmotionImages, getDataLoaders
from genderData import GenderImages
from ageData import AgeImages


def main():
    # Initialize DataSets
    dataset = EmotionImages()
    genderDataset = GenderImages()
    ageDataset = AgeImages()

    # Setup Databases
    dataset.setup()
    genderDataset.setup()
    ageDataset.setup()

    # Choose to Either Clean or Visualize Dataset
    models = []
    if len(sys.argv) > 1:
        if sys.argv[1] == "--display":
            dataset.plotVisuals()
            return
        elif sys.argv[1] == "--variant1":
            models.append(CNNVariant1())
        elif sys.argv[1] == "--base":
            models.append(CNNModel())
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
        models.append(CNNVariant2())

    # Gather Inputs
    trainDataloader, tesDataloader, validationDataloader = (
        getDataLoaders(dataset.getImageSplitDataset()))  # Emotion

    maleTrainDataloader, maleTesDataloader, maleValidationDataloader = (
        getDataLoaders(genderDataset.getMaleImageSplitDataset()))  # Male

    femaleTrainDataloader, femaleTesDataloader, femaleValidationDataloader = getDataLoaders(
        genderDataset.getFemaleImageSplitDataset())  # Female

    youngTrainDataloader, youngTesDataloader, youngValidationDataloader = getDataLoaders(
        ageDataset.getYoungImageSplitDataset())  # Young

    middleTrainDataloader, middleTesDataloader, middleValidationDataloader = getDataLoaders(
        ageDataset.getMiddleImageSplitDataset())  # Middle

    seniorTrainDataloader, seniorTesDataloader, seniorValidationDataloader = getDataLoaders(
        ageDataset.getSeniorImageSplitDataset())  # Senior

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveFile: str = ''

    # Train All Models
    for model in models:
        if isinstance(model, CNNModel):
            saveFile: str = "model.pth"
        elif isinstance(model, CNNVariant1):
            saveFile: str = "variant1.pth"
        elif isinstance(model, CNNVariant2):
            saveFile: str = "variant2.pth"

        # Training Main Datasets
        """ 
        trainCNN(dataLoader=[trainDataloader, tesDataloader, validationDataloader], model=model,
              device=device, savePath=os.path.join(dataset.getDataDirectory(), "bin", saveFile))
        """

        # Train for Biasing
        """
        trainCNN(dataLoader=[maleTrainDataloader, maleTesDataloader, maleValidationDataloader], model=model,
                 device=device, savePath=os.path.join(genderDataset.getDataDirectory(), "bin", saveFile))
        """


        # Train & Test CNN Model with K-Fold
        """
        trainCNNWithKFold(dataLoader=[trainDataloader, tesDataloader, validationDataloader], model=model,
                          device=device, savePath=os.path.join(dataset.getDataDirectory(), "bin", saveFile))
        """

    if len(models) > 0:
        random_image(tesDataloader, models)
    else:
        random_image(tesDataloader, [CNNVariant2()])

    # Display Dataset
    plt.show()


if __name__ == '__main__':
    main()
