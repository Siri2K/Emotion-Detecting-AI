import os
import platform
import random

from typing import Dict, List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms, ToTensor


def cleanImage(path) -> Image:
    """
        Resize Images and Saving Image
        https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil/13211834#13211834

        grayImage was setup using https://stackoverflow.com/a/3823822
        resizedImage wasse tup using https://stackoverflow.com/a/13211834

        :param path:
        :return:
    """

    image = Image.open(path)  # Open Image File
    if image.size != (336, 336) or image.mode != 'L':
        image = image.convert('L').resize((336, 336), Image.LANCZOS)
        image.save(path, optimize=True, quality=95)
    return image


def gatherRGBOfImages(images: List[Image.Image]) -> List[int]:
    """
        Formula for intensity was acquired using ChatGPT
    :param images:
    :return:
    """
    # Gather the Intensity of every images
    intensity: List[int] = []
    for image in images:
        intensity.append(int(sum(image.getdata()) / (image.width * image.height)))
    return intensity


def getDataLoaders(database:dict):
    # Update Dataset
    batchSize: int = 32

    # Setup Dataset
    train_dataset = ImageDataset(database['train'])
    test_dataset = ImageDataset(database['test'])
    validation_dataset = ImageDataset(database['validation'])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batchSize, shuffle=False)

    return train_dataloader, test_dataloader, validation_dataloader


class EmotionImages:
    # Constructor
    def __init__(self):
        self.__dataDirectory: str = ''
        self.__imageDataSet: Dict[str, List[Image.Image]] = {}
        self.__imageSplitDataSet: Dict[str, Dict[str, List[Image.Image]]] = {}
        self.__fileDataSet: Dict[str, List[str]] = {}

        # Initialize
        self.saveDataDirectory()

    # Set and Getter
    def setDataDirectory(self, dataDirectory: str):
        self.__dataDirectory = dataDirectory

    def setImageDataset(self, dataset: Dict[str, List[Image.Image]]):
        self.__imageDataSet = dataset

    def setImageSplitDataset(self, dataset: Dict[str, Dict[str, List[Image.Image]]]):
        self.__imageSplitDataSet = dataset

    def setFileDataset(self, dataset: Dict[str, List[str]]):
        self.__fileDataSet = dataset

    def getDataDirectory(self) -> str:
        return self.__dataDirectory

    def getImageDataset(self) -> Dict[str, List[Image.Image]]:
        return self.__imageDataSet.copy()

    def getImageSplitDataset(self) -> Dict[str, Dict[str, List[Image.Image]]]:
        return self.__imageSplitDataSet.copy()

    def getFileDataset(self) -> Dict[str, List[str]]:
        return self.__fileDataSet.copy()

    # Role
    def setup(self):
        self.setupFullData()
        self.splitData(self.getImageDataset())

    def saveDataDirectory(self):
        # Save Data Directory
        projectDirectory: str = os.path.dirname(os.path.abspath(__file__))
        if projectDirectory.endswith("src"):
            desiredDirectory = os.path.join(os.path.dirname(projectDirectory))
        else:
            desiredDirectory = os.path.join(projectDirectory)

        # Initialize
        self.setDataDirectory(desiredDirectory)

        # Delete Unusable
        del projectDirectory
        del desiredDirectory

    def setupFullData(self):
        """
            for root, directory, files in os.walk(desiredDirectory)
                for file in files
                .
                .
                .
            is done using  https://stackoverflow.com/a/2909998 and ChatGPT to iterate between files.
            Specifically the for loop condition, not the implementation within the loop
        :return:
        """

        # Initialize Data
        imageDataDict: Dict[str, List[Image.Image]] = {}
        fileDataDict: Dict[str, List[str]] = {}
        imageDataSet: List[Image.Image] = []
        fileDataSet: List[str] = []

        # Gather Images and File List
        for root, directory, files in os.walk(os.path.join(self.getDataDirectory(), "resources")):
            for file in files:
                # Gather Files & Images
                if file.endswith(".jpg") or file.endswith(".png"):
                    path: str = os.path.join(root, file)

                    # Setup File
                    fileDataSet.append(path)

                    # Setup Emotion Datasets
                    image: Image = cleanImage(path)
                    imageDataSet.append(image)

            # Store Them
            if len(fileDataSet) > 0 and len(imageDataSet) > 0:
                folder = root.split("\\")[-1] if platform.system() == "Windows" else root.split("/")[-1]
                fileDataDict[folder] = fileDataSet.copy()
                imageDataDict[folder] = imageDataSet.copy()

                # Clear Data
                fileDataSet.clear()
                imageDataSet.clear()

        # Setup Data
        self.setFileDataset(fileDataDict.copy())
        self.setImageDataset(imageDataDict.copy())

        # Delete Ununsed
        del imageDataDict
        del fileDataDict
        del imageDataSet
        del fileDataSet

    def splitData(self, data:Dict[str, List[Image.Image]]):
        """
        train_test_split without shuffling was taken by:
        https://stackoverflow.com/questions/43838052/how-to-get-a-non-shuffled-train-test-split-in-sklearn

        :return:
        """

        # Initialize
        splitData: dict = {
            'train': {
                'Angry': [],
                'Focused': [],
                'Happy': [],
                'Neutral': []
            },
            'test': {
                'Angry': [],
                'Focused': [],
                'Happy': [],
                'Neutral': []
            },
            'validation': {
                'Angry': [],
                'Focused': [],
                'Happy': [],
                'Neutral': []
            }
        }

        # Split Data
        for key, images in data.items():
            # Full image list setup to 70:30. Training is the 70%
            train, validationAndTest = train_test_split(
                images,
                test_size=0.3,
                shuffle=False
            )

            # 30% non training image list setup to 50:50 for validation and test
            validation, test = train_test_split(
                validationAndTest,
                test_size=0.5,
                shuffle=False
            )

            # Store Split Data
            splitData['train'][key] = train
            splitData['validation'][key] = validation
            splitData['test'][key] = test

        # Setup and delete unused
        self.setImageSplitDataset(splitData.copy())
        del splitData
        del data

    def plotVisuals(self):
        self.plotSampleImageGrid()
        self.plotClassDistribution()
        self.plotPixelIntensityDistributionClass()
        self.plotPixelIntensityDistributionClassForSample()
        plt.show()

    def gatherSampleImagesIndexes(self) -> List[list]:
        # Initialize
        images: dict = self.getImageDataset()
        keys: list = list(images.keys())
        sampleImages: List[list] = []

        # Gather Sample
        for i in range(15):
            chosenKey: str = random.choice(keys)
            sampleImages.append(
                [
                    chosenKey,
                    random.randint(0, len(images[chosenKey]) - 1)
                ]
            )
        return sampleImages

    def plotSampleImageGrid(self):
        # Gather Inputs
        indexList: List = self.gatherSampleImagesIndexes()
        images: dict = self.getImageDataset()

        # Initialize Grid
        nRows: int = 3
        nColumns: int = 5

        # Setup Figure
        figure, axImages = plt.subplots(nRows, nColumns, figsize=(6, 8))
        for i in range(nRows):
            for j in range(nColumns):
                indexPair = indexList.pop()
                axImages[i, j].imshow(images.get(indexPair[0])[indexPair[1]], cmap='gray')
                axImages[i, j].axis('off')
                axImages[i, j].set_title(indexPair[0])

        figure.suptitle("Sample Image Grid", fontsize=16)
        figure.tight_layout()

    def plotClassDistribution(self):
        """
            the graph was set up using
            https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/
        :return:
        """

        # Initialize Variables
        images = self.getImageDataset()

        # Setup X-Y
        X: list = list(images.keys())
        X_Axis: np.ndarray = np.arange(len(X))
        Y = []
        for key in X:
            Y.append(len(images.get(key)))

        # Setup Plot
        plt.figure()
        plt.bar(X_Axis, Y, 0.4)
        for i in range(len(X)):
            plt.text(i, Y[i] + 5, str(Y[i]), ha='center', va='bottom')
        plt.xticks(X_Axis, X)
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.title("Number of Images in each class")
        plt.ylim(0, 600)

    def plotPixelIntensityDistributionClassForSample(self):
        # Gather Inputs
        indexList: List = self.gatherSampleImagesIndexes()
        imageDict: dict = self.getImageDataset()
        imageList: List = []

        # Gather Pixel Distribution
        for key, index in indexList:
            imageList.append(
                imageDict.get(key)[index]
            )

        # Setup Histogram
        pixel_values = gatherRGBOfImages(imageList)

        plt.figure()
        plt.hist(pixel_values, alpha=0.7, edgecolor='black')

        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Histogram for Pixel Intensity of Sample Images')

    def plotPixelIntensityDistributionClass(self):
        """
            # from GeekforGeeks
                https: // www.geeksforgeeks.org / opencv - python - program - analyze - image - using - histogram /
                alternative way to find histogram of an image
                    plt.hist(img.ravel(),256,[0,256])
                    plt.show()

            :return:
        """

        # Initialize Data
        imageDict: dict = self.getImageDataset()

        # Gather Pixel Distribution
        for key, images in imageDict.items():
            pixel_values = gatherRGBOfImages(images)

            plt.figure()
            plt.hist(pixel_values, alpha=0.7, edgecolor='black')

            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.title(f'Histogram for Pixel Intensity of {key} Images')


class ImageDataset(Dataset):
    def __init__(self, dataDict: Dict[str, List[Image.Image]]):
        self.data = []
        for label, images in dataDict.items():
            for image in images:
                transformed_image = ToTensor()(image)
                self.data.append({'image': transformed_image, 'label': label})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['label']
        return image, label

    def getDataSize(self):
        return self.__len__()

    def getData(self, idx):
        image, label = self.__getitem__(idx)
        return image, label

    def getAllLabels(self) -> List[str]:
        return list(set([data['label'] for data in self.data]))
