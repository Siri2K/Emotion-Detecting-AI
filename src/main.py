import sys

from train import CNNModel, CNNVariant1, CNNVariant2, EmotionImages, Train


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
    model: CNNModel = CNNModel()  # Original Model
    model.setup()

    variant1: CNNVariant1 = CNNVariant1()  # Variant1
    variant1.setup([1, 1, 1], [16, 32, 64])

    variant2: CNNVariant2 = CNNVariant2()  # Variant 2
    variant2.setup(4)

    # Train Models
    trainModel: Train = Train(dataset, model).train()
    trainVariant1: Train = Train(dataset, variant1).train()
    trainVariant2: Train = Train(dataset, variant2).train()


if __name__ == '__main__':
    main()
