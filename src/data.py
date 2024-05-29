import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from typing import List
from PIL import Image



def savedImages(imageList: List[List[Image.Image]], fileLists: List[List[str]]):
    """
    For Loop setup using ChatGPT image.save() was setup with the help of
    https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil/13211834#13211834

    :param imageList:
    :param fileLists:
    :return:
    """

    for image_list, file_list in zip(imageList, fileLists):
        for image, file_path in zip(image_list, file_list):
            image.save(file_path, optimize=True, quality=95) # Save Images and Optimize Size


class EmotionImages:
    # Constructor
    def __init__(self):
        super().__init__()
        # self.emotions: int = 0
        self.__images: List[List[Image.Image]] = []
        self.__file: List[List[str]] = []

    # Set & Getter
    def setImages(self, image: List[List[Image.Image]]):
        self.__images = image

    def setFiles(self, file: List[List[str]]):
        self.__file = file

    def getImages(self) -> List[List[Image.Image]]:
        return self.__images.copy()

    def getFiles(self) -> List[List[str]]:
        return self.__file.copy()

    # Roles
    def initialize(self):
        self.readImages() # Gather Image and File Path from every file
        self.cleanImages() # GrayScale and Resize Files
        savedImages(self.getImages(), self.getFiles()) # Save new images into their respective files

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

            grayImage was setup using https://stackoverflow.com/a/3823822
            resizedImage was setup using https://stackoverflow.com/a/13211834
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
                resizedImage: Image.Image = grayImage.resize((336, 336), Image.LANCZOS) # Resize every pic to 336x336 and use Lanczos interpolation for quality
                newImages.append(resizedImage)
            newImageList.append(newImages)

        self.setImages(newImageList)
