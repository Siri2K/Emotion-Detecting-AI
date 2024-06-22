import os
import sys

from train import CNNModel, CNNVariant1, CNNVariant2
from train import plt
from train import torch
from train import trainCNN, trainCNNWithKFold, random_image
from data import EmotionImages


def main():
    # Initialize DataSet
    dataset = EmotionImages()
    dataset.setup()

    # Choose to Either Clean or Visualize Dataset
    models = []
    if len(sys.argv) > 1:
        if sys.argv[1] == "--display":
            dataset.plotVisuals()
            return
        elif sys.argv[1] == "--variant1":
            models.append(CNNVariant1())
        elif sys.argv[1] == "--base":
            models.append(CNNModel())
        elif sys.argv[1] == "--trainAll":
            models.append(CNNModel())
            models.append(CNNVariant1())
            models.append(CNNVariant2())
        elif sys.argv[1] == "--none":
            pass
        else:
            print("Invalid Command")
            print("Please Enter : python main.py or python main.py --display")
            return
    else:
        models.append(CNNVariant2())

    # Gather Inputs
    train_dataloader, test_dataloader, validation_dataloader = dataset.getDataLoaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saveFile: str = ''

    # Train All Models
    for model in models:
        if isinstance(model, CNNModel):
            saveFile: str = "model.pth"
        elif isinstance(model, CNNVariant1):
            saveFile: str = "variant1.pth"
        elif isinstance(model, CNNVariant2):
            saveFile: str = "variant2.pth"

        # Train & Test CNN Model with K-Fold
        trainCNNWithKFold(dataLoader=[train_dataloader, test_dataloader, validation_dataloader], model=model,
                          device=device,savePath=os.path.join(dataset.getDataDirectory(), "bin", saveFile))

    if len(models) > 0:
        random_image(test_dataloader, models)
    else:
        random_image(test_dataloader, [CNNVariant2()])

    # Display Dataset
    plt.show()


if __name__ == '__main__':
    main()
