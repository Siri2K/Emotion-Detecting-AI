import os
import platform
from typing import Dict, List

from PIL import Image
from sklearn.model_selection import train_test_split

import emotionData

from emotionData import EmotionImages
from emotionData import cleanImage


class GenderImages(EmotionImages):
    def __init__(self):
        super().__init__()

        self.__femaleImageDataSet: Dict[str, List[Image.Image]] = {}
        self.__maleImageDataSet: Dict[str, List[Image.Image]] = {}

        self.__femaleImageSplitDataSet: Dict[str, Dict[str, List[Image.Image]]] = {}
        self.__maleImageSplitDataSet: Dict[str, Dict[str, List[Image.Image]]] = {}

        self.__femaleFileDataSet: Dict[str, List[str]] = {}
        self.__maleFileDataSet: Dict[str, List[str]] = {}

        # Initialize
        self.saveDataDirectory()
        self.setup()

    # Set and Getter
    def setFemaleImageDataset(self, dataset: Dict[str, List[Image.Image]]):
        self.__femaleImageDataSet = dataset

    def setMaleImageDataset(self, dataset: Dict[str, List[Image.Image]]):
        self.__maleImageDataSet = dataset

    def setFemaleImageSplitDataset(self, dataset: Dict[str, Dict[str, List[Image.Image]]]):
        self.__femaleImageSplitDataSet = dataset

    def setMaleImageSplitDataset(self, dataset: Dict[str, Dict[str, List[Image.Image]]]):
        self.__maleImageSplitDataSet = dataset

    def setFemaleFileDataset(self, dataset: Dict[str, List[str]]):
        self.__femaleFileDataSet = dataset

    def setMaleFileDataset(self, dataset: Dict[str, List[str]]):
        self.__maleFileDataSet = dataset

    def getFemaleImageDataset(self) -> Dict[str, List[Image.Image]]:
        return self.__femaleImageDataSet.copy()

    def getMaleImageDataset(self) -> Dict[str, List[Image.Image]]:
        return self.__maleImageDataSet.copy()

    def getFemaleImageSplitDataset(self) -> Dict[str, Dict[str, List[Image.Image]]]:
        return self.__femaleImageSplitDataSet.copy()

    def getMaleImageSplitDataset(self) -> Dict[str, Dict[str, List[Image.Image]]]:
        return self.__maleImageSplitDataSet.copy()

    def getFemaleFileDataset(self) -> Dict[str, List[str]]:
        return self.__femaleFileDataSet.copy()

    def getMaleFileDataset(self) -> Dict[str, List[str]]:
        return self.__maleFileDataSet.copy()

    def setup(self):
        self.setupFullData()
        self.splitData(self.getMaleImageDataset())
        self.splitData(self.getFemaleImageDataset())

    def setupFullData(self):
        # Initialize Data
        femaleImageDataDict: Dict[str, List[Image.Image]] = {}
        maleImageDataDict: Dict[str, List[Image.Image]] = {}

        femaleFileDataDict: Dict[str, List[str]] = {}
        maleFileDataDict: Dict[str, List[str]] = {}

        femaleImageDataSet: List[Image.Image] = []
        maleImageDataSet: List[Image.Image] = []

        femaleFileDataSet: List[str] = []
        maleFileDataSet: List[str] = []

        # Gather Images and File List
        for root, directory, files in os.walk(os.path.join(self.getDataDirectory(), "resources")):
            for file in files:
                # Gather Files & Images
                if file.endswith(".jpg") or file.endswith(".png"):
                    # Get Full Path Ready to Save
                    path: str = os.path.join(root, file)

                    # Split File into List and Get the Gender from FileName
                    gender: str = file.split('_')[-2]

                    # Setup Files and Dataset
                    if gender == "Male":
                        maleFileDataSet.append(path)
                        image: Image = cleanImage(path)
                        maleImageDataSet.append(image)
                    elif gender == "Female":
                        femaleFileDataSet.append(path)
                        image: Image = cleanImage(path)
                        femaleImageDataSet.append(image)
                    else:
                        pass

            # Store Image Dataset
            if len(maleFileDataSet) > 0 and len(maleImageDataSet):
                folder = root.split("\\")[-1] if platform.system() == "Windows" else root.split("/")[-1]
                maleFileDataDict[folder] = maleFileDataSet.copy()
                maleImageDataDict[folder] = maleImageDataSet.copy()

                # Clear Data
                maleFileDataSet.clear()
                maleImageDataSet.clear()

            if len(femaleFileDataSet) > 0 and len(femaleImageDataSet):
                folder = root.split("\\")[-1] if platform.system() == "Windows" else root.split("/")[-1]
                femaleFileDataDict[folder] = femaleFileDataSet.copy()
                femaleImageDataDict[folder] = femaleImageDataSet.copy()

                # Clear Data
                femaleFileDataSet.clear()
                femaleImageDataSet.clear()

        # Setup Data
        self.setFemaleFileDataset(femaleFileDataDict)
        self.setMaleFileDataset(maleFileDataDict)

        self.setFemaleImageDataset(femaleImageDataDict)
        self.setMaleImageDataset(maleImageDataDict)

        # Delete Unused
        del femaleImageDataDict
        del maleImageDataDict

        del femaleFileDataDict
        del maleFileDataDict

        del femaleImageDataSet
        del maleImageDataSet

        del femaleFileDataSet
        del maleFileDataSet

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
        if data == self.getFemaleImageDataset():
            self.setFemaleImageSplitDataset(splitData.copy())
        elif data == self.getMaleImageDataset():
            self.setMaleImageSplitDataset(splitData.copy())

        del splitData
        del data
