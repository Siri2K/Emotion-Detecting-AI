import sys
from data import EmotionImages
from cnn import CNN


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
    model: CNN = CNN() # Original Model
    model.initialize([3,32,32,64], outChannel=[32,32,64,64], kernelSize=3)

    variant1: CNN = CNN() # Variant1
    variant1.initialize([1, 1, 1], [16, 32, 64], 3)


if __name__ == '__main__':
    main()
