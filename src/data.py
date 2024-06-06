import os
import platform

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import torch.utils.data as td

from typing import List
from PIL import Image
from sklearn.model_selection import train_test_split


def savedImages(imageList: List[List[Image.Image]], fileLists: List[List[str]]):
    """
    For Loop setup using ChatGPT image.save() was set up with the help of
    https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil/13211834#13211834

    :param imageList:
    :param fileLists:
    :return:
    """

    for image_list, file_list in zip(imageList, fileLists):
        for image, file_path in zip(image_list, file_list):
            image.save(file_path, optimize=True, quality=95)  # Save Images and Optimize Size


def display():
    plt.show()


def gatherRGBOfImages(images: List[Image.Image]):
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


class EmotionImages:
    # Constructor
    def __init__(self):
        super().__init__()
        # self.emotions: int = 0
        self.__sampleImages: List[Image.Image] = []
        self.__images: List[List[Image.Image]] = []
        self.__file: List[List[str]] = []
        self.__dataset: dict = {
            'train': {},
            'test': {},
            'validation': {}
        }
        self.__dataLoader: dict = {
            'train': {},
            'test': {},
            'validation': {}
        }

    # Set & Getter
    def setImages(self, image: List[List[Image.Image]]):
        self.__images = image

    def setSampleImages(self, sampleImage: List[Image.Image]):
        self.__sampleImages = sampleImage

    def setFiles(self, file: List[List[str]]):
        self.__file = file

    def setDataset(self, dataset: dict):
        self.__dataset = dataset

    def setDataLoader(self, dataLoader: dict):
        self.__dataLoader = dataLoader

    def getImages(self) -> List[List[Image.Image]]:
        return self.__images.copy()

    def getSampleImages(self) -> List[Image.Image]:
        return self.__sampleImages.copy()

    def getFiles(self) -> List[List[str]]:
        return self.__file.copy()

    def getDataset(self) -> dict:
        return self.__dataset.copy()

    def getDataLoader(self) -> dict:
        return self.__dataLoader.copy()

    # Roles
    def initialize(self):
        self.readImages()  # Gather Image and File Path from every file
        self.splitData()

    def readImages(self):
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

        # Get Desired Directory
        projectDirectory: str = os.path.dirname(os.path.abspath(__file__))
        if projectDirectory.endswith("src"):
            desiredDirectory = os.path.join(os.path.dirname(projectDirectory), "resources")
        else:
            desiredDirectory = os.path.join(projectDirectory, "resources")

        # Obtain All Image files
        savedFileList: List[List[str]] = []
        imageList: List[List[Image.Image]] = []
        for root, directory, files in os.walk(desiredDirectory):
            savedFile: List[str] = []
            images: List[Image.Image] = []
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    path: str = os.path.join(root, file)
                    savedFile.append(path)
                    images.append(Image.open(path))

            if len(savedFile) > 0:
                savedFileList.append(savedFile)

            if len(images) > 0:
                imageList.append(images)

        # Store File & Image List
        self.setFiles(savedFileList)
        self.setImages(imageList)

    def cleanImages(self):
        """
            grayImage: Image.Image = face.convert("L")  # L means gray coloring
            resizedImage: Image.Image = grayImage.resize((336, 336), Image.LANCZOS)

            grayImage was se tup using https://stackoverflow.com/a/3823822
            resizedImage was se tup using https://stackoverflow.com/a/13211834
        :return:
        """

        # Get Images
        imageList: List[List[Image.Image]] = self.getImages()
        newImageList: List[List[Image.Image]] = []

        # Generate Images
        for faceList in imageList:
            newImages: List[Image.Image] = []
            for face in faceList:
                grayImage: Image.Image = face.convert("L")  # GRayScale Images using "L" command

                # Lanczos interpolation for quality
                # Resize every pic to 336x336 and use
                resizedImage: Image.Image = grayImage.resize((336, 336), Image.LANCZOS)

                newImages.append(resizedImage)
            newImageList.append(newImages)

        # Save New Images and Files
        self.setImages(newImageList)
        savedImages(self.getImages(), self.getFiles())  # Save new images into their respective files

    def gatherImageIndexes(self) -> List[List[int]]:
        # Initialize data
        images: List[List[Image]] = self.getImages()

        # Get Indexes of images to display
        indexList: List[List[int]] = []
        indexPair: List[int] = []
        for i in range(15):
            # Get Pairs
            indexPair.append(rnd.randrange(start=0, stop=len(images)))
            indexPair.append(rnd.randrange(start=0, stop=len(images[indexPair[0]])))

            # Check if Pair Repeats & Get A New Pair
            while indexList.count(indexPair) > 0:
                indexPair.clear()
                indexPair.append(rnd.randrange(start=0, stop=len(images)))
                indexPair.append(rnd.randrange(start=0, stop=len(images[indexPair[0]])))

            # Add them to list and clear
            indexList.append(indexPair.copy())
            indexPair.clear()

        return indexList

    def plotImageGrid(self, indexList: List[List[int]]):
        """
        Lines 179 and Lines 180 were set up using ChatGPT. Mainly to separate titles between main plot and subplot
        axImage setup was configured with the help of ChatGPT and
        https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

        :param indexList:
        :return:
        """

        # Initialize data
        sampleImages: List[Image] = []
        images: List[List[Image]] = self.getImages()

        # Setup Grid Images
        nRows: int = 3
        nColumns: int = 5

        figure, axImages = plt.subplots(nRows, nColumns, figsize=(6, 8))
        for i in range(nRows):
            for j in range(nColumns):
                # Get Index Pair
                indexPair: List[int] = indexList.pop()

                # Get Image title from Filepath
                if platform.system() == "Windows":
                    sampleFilepath = self.getFiles()[indexPair[0]].pop().split("\\")
                else:
                    sampleFilepath = self.getFiles()[indexPair[0]].pop().split("/")

                sampleImages.append(images[indexPair[0]][indexPair[1]])
                axImages[i, j].imshow(images[indexPair[0]][indexPair[1]], cmap='gray')
                axImages[i, j].axis('off')
                axImages[i, j].set_title(sampleFilepath[-2])

        # Setup Plots
        self.setSampleImages(sampleImages)
        figure.suptitle("Sample Image Grid", fontsize=16)
        figure.tight_layout()

    # Pixel Intensity Distribution per Class
    def pixelIntensityDistributionClass(self):

        image_List = self.getImages()

        index = 0

        for image_group in image_List:
            # from GeekforGeeks
            """ https: // www.geeksforgeeks.org / opencv - python - program - analyze - image - using - histogram /
            # alternative way to find histogram of an image 
                plt.hist(img.ravel(),256,[0,256]) 
                plt.show() 
            """
            # Get Image title from Filepath
            if platform.system() == "Windows":
                sampleFilepath = self.getFiles()[index].pop().split("\\")
            else:
                sampleFilepath = self.getFiles()[index].pop().split("/")

            pixel_values = gatherRGBOfImages(image_group)
            plt.figure()
            plt.hist(pixel_values, bins=256, alpha=0.7, edgecolor='black')

            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.title(f'Histogram for Pixel Intensity of {sampleFilepath[-2]} Images')
            index += 1

    def plotClassDistribution(self):
        """
            the graph was set up using
            https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/
        :return:
        """

        # Initialize Variables
        images = self.getImages()
        X = []
        Y = []

        # Setup X and Y
        for i in range(len(images)):
            if platform.system() == "Windows":
                folder = self.getFiles()[i].pop().split("\\")
            else:
                folder = self.getFiles()[i].pop().split("/")
            X.append(folder[-2])
            Y.append(len(images[i]))

        X_axis = np.arange(len(X))

        plt.figure()
        plt.bar(X_axis, Y, 0.4)

        for i in range(len(X)):
            plt.text(i, Y[i] + 5, str(Y[i]), ha='center', va='bottom')

        plt.xticks(X_axis, X)
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.title("Number of Images in each class")
        plt.ylim(0, 600)

    def plotPixelIntensityForSample(self):
        # Gather Red, Green , Blue Intensities
        intensity = gatherRGBOfImages(self.getSampleImages())

        # Create Histograms for each Color Intensity
        plt.figure()
        plt.hist(intensity, alpha=0.7, edgecolor='black')
        plt.title("Sample Image Pixel Intensity", fontsize=16)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Average Frequency")
        plt.tight_layout()

    def plotVisuals(self):
        self.plotImageGrid(self.gatherImageIndexes())
        self.plotPixelIntensityForSample()
        self.pixelIntensityDistributionClass()
        self.plotClassDistribution()
        display()

    '''
    # split the dataset into training, testing and validation sets
    def splitData(self, random_state=42):
        total_images = len(self.getImages())
        train_size = int(0.7 * total_images)
        test_size = int(0.15 * total_images)
        validation_size = total_images - train_size - test_size
        self.train_images = self.getImages()[:train_size]
        self.test_images = self.getImages()[train_size:train_size + test_size]
        self.validation_images = self.getImages()[train_size + test_size:]
        self.train_labels = [i for i in range(len(self.train_images))]
        self.test_labels = [i for i in range(len(self.test_images))]
        self.validation_labels = [i for i in range(len(self.validation_images))]
        self.train_labels = torch.tensor(self.train_labels)
        self.test_labels = torch.tensor(self.test_labels)
        self.validation_labels = torch.tensor(self.validation_labels)
        self.train_images = torch.tensor(self.train_images)
        self.test_images = torch.tensor(self.test_images)
        self.validation_images = torch.tensor(self.validation_images)
    '''

    def splitData(self):
        """
        train_test_split setup was inspired by lab 5:
        https://moodle.concordia.ca/moodle/pluginfile.php/6908483/mod_resource/content/9/lab05-sol.pdf

        train_test_split without shuffling was taken by:
        https://stackoverflow.com/questions/43838052/how-to-get-a-non-shuffled-train-test-split-in-sklearn

        :return:
        """

        # Gather Image and FileList
        imageSet = self.getImages()
        fileSet = self.getFiles()
        dataset = self.getDataset()

        # Split Files with scikit-learn
        for images, files in zip(imageSet, fileSet):
            # Get Image title from Filepath
            if platform.system() == "Windows":
                emotion = files.pop().split("\\")[-2]
            else:
                emotion = files.pop().split("/")[-2]

            # Split into Training Testing & Validation (Training : 70%, Testing: 15%, Validation: 15%)
            trainImages, testAndValidationImages = train_test_split(
                images,
                test_size=0.3,
                shuffle=False
            )  # Training: 70%, Test And validation Combined = 30%

            testImages, validationImages = train_test_split(
                testAndValidationImages,
                test_size=0.5,
                shuffle=False
            )  # Test: 50%, Validation Combined = 50%

            # Store within Dataset
            dataset['train'][emotion] = trainImages
            dataset['test'][emotion] = testImages
            dataset['validation'][emotion] = validationImages

        # Store Final Result
        self.setDataset(dataset)

    def dataLoader(self):
        dataset = self.getDataset()
        dataLoader: dict = {}

        # Hyper-parameters
        batch_size = 4

        # Setup DataLoader
        dataLoader['train'] = td.DataLoader(dataset['train'], batch_size=batch_size,
                                            shuffle=False, pin_memory=True)
        dataLoader['test'] = td.DataLoader(dataset['test'], batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        dataLoader['validation'] = td.DataLoader(dataset['validation'], batch_size=batch_size,
                                                 shuffle=False, pin_memory=True)

        self.setDataLoader(dataLoader)

        for keys in self.getDataLoader()['train']:
            print(keys)