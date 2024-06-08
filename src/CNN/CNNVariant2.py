import torch.nn as nn

from src.CNN.CNNModel import CNNModel, OrderedDict, List


class CNNVariant2(CNNModel):
    def __init__(self):
        super(CNNVariant2, self).__init__()
        self.__layers = None
        self.__fcLayer = None

    def setup(self, kernelSize: int):
        inChannel: List[int] = [3, 32, 32, 64]
        outChannel: List[int] = [32, 32, 64, 64]
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
