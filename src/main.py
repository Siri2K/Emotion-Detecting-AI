import sys
import torch
import torch.nn as nn

from data import EmotionImages
from cnn import CNN


# Train Models
def trainModel(dataset: EmotionImages, model: CNN):
    num_epochs = 10
    batch_size = 4
    num_classes = 4
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(dataset.getDataLoader()['train'])
    loss_list = []
    acc_list = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataset.getDataLoader()['train']):  # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            if (i + 1) % 100 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1,
                                                                                          total_step, loss.item(),
                                                                                          (correct / total) * 100))


def main():
    # Initialize DataSet
    dataset: EmotionImages = EmotionImages()
    dataset.initialize()

    # Choose to Either Clean or Visualize Dataset
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clean":
            dataset.cleanImages()
            print("Dataset Cleaned")
            return
        elif sys.argv[1] == "--display":
            dataset.plotVisuals()
            return
        else:
            print("Invalid Command")
            print("Please Enter : python main.py or python main.py --clean")
            return

    # Initialize CNNs
    model: CNN = CNN()  # Original Model
    model.initialize([3, 32, 32, 64], outChannel=[32, 32, 64, 64], kernelSize=3)

    variant1: CNN = CNN()  # Variant1
    variant1.initialize([1, 1, 1], [16, 32, 64], 3)

    variant2: CNN = CNN()  # Variant 2
    variant2.initialize([3, 32, 32, 64], [32, 32, 64, 64], 4)

    # Train Models
    trainModel(dataset, model)


if __name__ == '__main__':
    main()
