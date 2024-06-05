import sys
from data import EmotionImages


def main():
    # Initialize DataSet
    dataset: EmotionImages = EmotionImages()
    dataset.initialize()

    # Choose to Either Clean or Visualize Dataset
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clean":
            dataset.cleanImages()
            print("Dataset Cleaned")
        elif sys.argv[1] == "--display":
            dataset.plotVisuals()
        else:
            print("Invalid Command")
            print("Please Enter : python main.py or python main.py --clean")
    else:
        dataset.plotVisuals()


if __name__ == '__main__':
    main()
