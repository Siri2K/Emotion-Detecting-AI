# Imports
from collections import OrderedDict
from typing import List

import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.__layers = None
        self.__fcLayer = None

    def setLayers(self, layers: nn.Sequential):
        self.__layers = layers

    def setFCLayers(self, fcLayer: nn.Sequential):
        self.__fcLayer = fcLayer

    def getLayers(self) -> nn.Sequential:
        return self.__layers

    def getFCLayers(self) -> nn.Sequential:
        return self.__fcLayer

    # Roles
    def setup(self):
        """
        - Layers and FC Layers setup with the help of:
          https://moodle.concordia.ca/moodle/pluginfile.php/6908445/mod_resource/content/3/a2.pdf

        - Formula to predict Convolution Layer size obtained from :
          https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        """

        inChannel: List[int] = [3, 32, 32, 64]
        outChannel: List[int] = [32, 32, 64, 64]
        kernelSize: int = 3
        padding: int = 1
        stride: int = 1

        # Create Sequential Layer
        layers: OrderedDict = OrderedDict()
        fcLayers: OrderedDict = OrderedDict()

        # Create Arguments for Sequential Layers
        for i in range(len(outChannel)):
            layers[f'conv{i}'] = nn.Conv2d(in_channels=inChannel[i], out_channels=outChannel[i], kernel_size=kernelSize,
                                           padding=1)
            layers[f'batch{i}'] = nn.BatchNorm2d(outChannel[i])
            layers[f'relu{i}'] = nn.LeakyReLU(inplace=True)
            layers[f'maxpool{i}'] = nn.MaxPool2d(kernel_size=2, stride=2)

        # Create Arguments for FC Layers
        for i in range(len(outChannel)):
            # Apply Formula to find Convolution Sizes
            layerSize: int = int((inChannel[i] - kernelSize + 2 * padding) / stride) + 1
            fcLayers[f'Linear{i}'] = nn.Linear(
                in_features=32 * layerSize * layerSize,
                out_features=4)

        # Setup Convolution and FC Layer
        self.setLayers(nn.Sequential(layers))
        self.setFCLayers(nn.Sequential(fcLayers))

    def forward(self, x) -> nn.Sequential:
        """
        - Function setup with the help of :
          https://moodle.concordia.ca/moodle/pluginfile.php/6908445/mod_resource/content/3/a2.pdf

        :param x:
        :return:
        """

        layers = self.getLayers()
        fcLayers = self.getFCLayers()

        # Setup Layers
        x = layers(x)  # Convolution Layers
        x = x.view(x.size(0), -1)  # Flatten
        x = fcLayers(x)  # FC Layer

        return x
