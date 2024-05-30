import os

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

from sklearn.model_selection import train_test_split
from typing import List
from PIL import Image


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
    '''
        Formula for intensity was acquired using ChatGPT
    :param images:
    :return:
    '''

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

    # Set & Getter
    def setImages(self, image: List[List[Image.Image]]):
        self.__images = image

    def setSampleImages(self, sampleImage: List[Image.Image]):
        self.__sampleImages = sampleImage

    def setFiles(self, file: List[List[str]]):
        self.__file = file

    def getImages(self) -> List[List[Image.Image]]:
        return self.__images.copy()

    def getSampleImages(self) -> List[Image.Image]:
        return self.__sampleImages.copy()

    def getFiles(self) -> List[List[str]]:
        return self.__file.copy()

    # Roles
    def initialize(self):
        self.readImages()  # Gather Image and File Path from every file
        self.cleanImages()  # GrayScale and Resize Files
        savedImages(self.getImages(), self.getFiles())  # Save new images into their respective files

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

        self.setImages(newImageList)

    def gatherImageIndexes(self) -> List[List[int]]:
        # Initialize data
        images: List[List[Image]] = self.getImages()

        # Get Indexes of images to display
        indexList: List[List[int]] = []
        indexPair: List[int] = []
        for i in range(15):
            # Get Pairs
            indexPair.append(rnd.randrange(start=0, stop=len(images) - 1))
            indexPair.append(rnd.randrange(start=0, stop=len(images[indexPair[0]]) - 1))

            # Add them to list and clear
            indexList.append(indexPair.copy())
            indexPair.clear()

        return indexList

    def plotImageGrid(self, indexList: List[List[int]]):
        '''
        Lines 179 and Lines 180 were setup using ChatGPT. Mainly to seperate titles between main plot and subplot
        axImage setup was configured with the help of ChatGPT and
        https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

        :param indexList:
        :return:
        '''

        # Initialize data
        sampleImages: List[Image] = []
        images: List[List[Image]] = self.getImages()

        # Setup Grid Images
        nRows: int = 3
        nColumns: int = 5
        title: str = ''

        figure, axImages = plt.subplots(nRows, nColumns, figsize=(6, 8))
        for i in range(nRows):
            for j in range(nColumns):
                indexPair: List[int] = indexList.pop()

                # Get Image Titles
                if indexPair[0] == 0:
                    title = "Angry"
                elif indexPair[0] == 1:
                    title = "Focused"
                elif indexPair[0] == 2:
                    title = "Happy"
                elif indexPair[0] == 3:
                    title = "Neutral"

                sampleImages.append(images[indexPair[0]][indexPair[1]])
                axImages[i, j].imshow(images[indexPair[0]][indexPair[1]], cmap='gray')
                axImages[i, j].axis('off')
                axImages[i, j].set_title(title)

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
            # Get Image Titles
            title = ''
            if index == 0:
                title = "Angry"
            elif index == 1:
                title = "Focused"
            elif index == 2:
                title = "Happy"
            elif index == 3:
                title = "Neutral"
            pixel_values = gatherRGBOfImages(image_group)
            plt.figure()
            plt.hist(pixel_values, bins=256, alpha=0.7, edgecolor='black')

            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.title(f'Histogram for Pixel Intensity of {title} Images')
            index += 1

    def classDistribution(self):
        """
            the graph was set up using https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/
        :return:
        """

        X = ['Angry', 'Focused', 'Happy', 'Neutral']
        Y = []

        # Setup Y
        for folders in self.getImages():
            Y.append(len(folders))

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

