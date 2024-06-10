import torch.nn as nn

from src.CNN.CNNModel import CNNModel


class CNNVariant1(CNNModel):
    def __init__(self):
        super(CNNVariant1, self).__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(32 * 19 * 19, 4)  # Assuming 336x336 input size and 4 output classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
