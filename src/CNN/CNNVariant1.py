import torch.nn as nn


class CNNVariant1(nn.Module):
    def __init__(self):
        super(CNNVariant1, self).__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.LeakyReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64*10*10, 4)  # Assuming 40x40 input size and 4 output classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
