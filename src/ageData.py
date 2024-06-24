import os
import platform
from typing import Dict, List

from PIL import Image
from sklearn.model_selection import train_test_split

import emotionData

from emotionData import EmotionImages
from emotionData import cleanImage


class AgeImages(EmotionImages):
    def __init__(self):
        super().__init__()

        self.__youngImageDataSet: Dict[str, List[Image.Image]] = {}
        self.__middleImageDataSet: Dict[str, List[Image.Image]] = {}
        self.__seniorImageDataSet: Dict[str, List[Image.Image]] = {}

        self.__youngImageSplitDataSet: Dict[str, Dict[str, List[Image.Image]]] = {}
        self.__middleImageSplitDataSet: Dict[str, Dict[str, List[Image.Image]]] = {}
        self.__seniorImageSplitDataSet: Dict[str, Dict[str, List[Image.Image]]] = {}

        self.__youngFileDataSet: Dict[str, List[str]] = {}
        self.__middleFileDataSet: Dict[str, List[str]] = {}
        self.__seniorFileDataSet: Dict[str, List[str]] = {}

        # Initialize
        self.saveDataDirectory()
        self.setup()

    # Set and Getter
    def setYoungImageDataset(self, dataset: Dict[str, List[Image.Image]]):
        self.__youngImageDataSet = dataset

    def setMiddleImageDataset(self, dataset: Dict[str, List[Image.Image]]):
        self.__middleImageDataSet = dataset

    def setSeniorImageDataset(self, dataset: Dict[str, List[Image.Image]]):
        self.__seniorImageDataSet = dataset

    def setYoungImageSplitDataset(self, dataset: Dict[str, Dict[str, List[Image.Image]]]):
        self.__youngImageSplitDataSet = dataset

    def setMiddleImageSplitDataset(self, dataset: Dict[str, Dict[str, List[Image.Image]]]):
        self.__middleImageSplitDataSet = dataset

    def setSeniorImageSplitDataset(self, dataset: Dict[str, Dict[str, List[Image.Image]]]):
        self.__seniorImageSplitDataSet = dataset

    def setYoungFileDataset(self, dataset: Dict[str, List[str]]):
        self.__youngFileDataSet = dataset

    def setMiddleFileDataset(self, dataset: Dict[str, List[str]]):
        self.__middleFileDataSet = dataset

    def setSeniorFileDataset(self, dataset: Dict[str, List[str]]):
        self.__seniorFileDataSet = dataset

    def getYoungImageDataset(self) -> Dict[str, List[Image.Image]]:
        return self.__youngImageDataSet.copy()

    def getMiddleImageDataset(self) -> Dict[str, List[Image.Image]]:
        return self.__middleImageDataSet.copy()

    def getSeniorImageDataset(self) -> Dict[str, List[Image.Image]]:
        return self.__seniorImageDataSet.copy()

    def getYoungImageSplitDataset(self) -> Dict[str, Dict[str, List[Image.Image]]]:
        return self.__youngImageSplitDataSet.copy()

    def getMiddleImageSplitDataset(self) -> Dict[str, Dict[str, List[Image.Image]]]:
        return self.__middleImageSplitDataSet.copy()

    def getSeniorImageSplitDataset(self) -> Dict[str, Dict[str, List[Image.Image]]]:
        return self.__seniorImageSplitDataSet.copy()

    def getYoungFileDataset(self) -> Dict[str, List[str]]:
        return self.__youngFileDataSet.copy()

    def getMiddleFileDataset(self) -> Dict[str, List[str]]:
        return self.__middleFileDataSet.copy()

    def getSeniorFileDataset(self) -> Dict[str, List[str]]:
        return self.__middleFileDataSet.copy()

    def setup(self):
        self.setupFullData()
        self.splitData(self.getYoungImageDataset())
        self.splitData(self.getMiddleImageDataset())
        self.splitData(self.getSeniorImageDataset())

    def setupFullData(self):
        """
        Removing Numbers from a string
        https://stackoverflow.com/questions/12851791/removing-numbers-from-string

        :return:
        """

        # Initialize Data
        youngImageDataDict: Dict[str, List[Image.Image]] = {}
        middleImageDataDict: Dict[str, List[Image.Image]] = {}
        seniorImageDataDict: Dict[str, List[Image.Image]] = {}

        youngFileDataDict: Dict[str, List[str]] = {}
        middleFileDataDict: Dict[str, List[str]] = {}
        seniorFileDataDict: Dict[str, List[str]] = {}

        youngImageDataSet: List[Image.Image] = []
        middleImageDataSet: List[Image.Image] = []
        seniorImageDataSet: List[Image.Image] = []

        youngFileDataSet: List[str] = []
        middleFileDataSet: List[str] = []
        seniorFileDataSet: List[str] = []

        # Gather Images and File List
        for root, directory, files in os.walk(os.path.join(self.getDataDirectory(), "resources")):
            for file in files:
                # Gather Files & Images
                if file.endswith(".jpg") or file.endswith(".png"):
                    # Get Full Path Ready to Save
                    path: str = os.path.join(root, file)

                    # Split File into List and Get the Gender from FileName
                    age: str = file.split('_')[-1]
                    age = age.split('.')[0]
                    age = ''.join([i for i in age if not i.isdigit()])

                    # Setup Files and Dataset
                    if age == "Young":
                        youngFileDataSet.append(path)
                        image: Image = cleanImage(path)
                        youngImageDataSet.append(image)
                    elif age == "Middle":
                        middleFileDataSet.append(path)
                        image: Image = cleanImage(path)
                        middleImageDataSet.append(image)
                    elif age == "Senior":
                        seniorFileDataSet.append(path)
                        image: Image = cleanImage(path)
                        seniorImageDataSet.append(image)
                    else:
                        pass

            # Store Image Dataset
            if len(youngFileDataSet) > 0 and len(youngImageDataSet):
                folder = root.split("\\")[-1] if platform.system() == "Windows" else root.split("/")[-1]
                youngFileDataDict[folder] = youngFileDataSet.copy()
                youngImageDataDict[folder] = youngImageDataSet.copy()

                # Clear Data
                youngFileDataSet.clear()
                youngImageDataSet.clear()

            if len(middleFileDataSet) > 0 and len(middleImageDataSet):
                folder = root.split("\\")[-1] if platform.system() == "Windows" else root.split("/")[-1]
                middleFileDataDict[folder] = middleFileDataSet.copy()
                middleImageDataDict[folder] = middleImageDataSet.copy()

                # Clear Data
                middleFileDataSet.clear()
                middleImageDataSet.clear()

            if len(seniorFileDataSet) > 0 and len(seniorImageDataSet):
                folder = root.split("\\")[-1] if platform.system() == "Windows" else root.split("/")[-1]
                seniorFileDataDict[folder] = seniorFileDataSet.copy()
                seniorImageDataDict[folder] = seniorImageDataSet.copy()

                # Clear Data
                seniorFileDataSet.clear()
                seniorImageDataSet.clear()

        # Setup Data
        self.setYoungFileDataset(youngFileDataDict)
        self.setMiddleFileDataset(middleFileDataDict)
        self.setSeniorFileDataset(seniorFileDataDict)

        self.setYoungImageDataset(youngImageDataDict)
        self.setMiddleImageDataset(middleImageDataDict)
        self.setSeniorImageDataset(seniorImageDataDict)

        # Delete Unused
        del youngImageDataDict
        del middleImageDataDict
        del seniorImageDataDict

        del youngFileDataDict
        del middleFileDataDict
        del seniorFileDataDict

        del youngImageDataSet
        del middleImageDataSet
        del seniorImageDataSet

        del youngFileDataSet
        del middleFileDataSet
        del seniorFileDataSet

    def splitData(self, data:Dict[str, List[Image.Image]]):
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
        if data == self.getYoungImageDataset():
            self.setYoungImageSplitDataset(splitData.copy())
        elif data == self.getMiddleImageDataset():
            self.setMiddleImageSplitDataset(splitData.copy())
        elif data == self.getSeniorImageDataset():
            self.setSeniorImageSplitDataset(splitData.copy())

        del splitData
        del data
