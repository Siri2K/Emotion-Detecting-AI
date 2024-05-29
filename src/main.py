from data import EmotionImages


def main():
    dataset: EmotionImages = EmotionImages()
    dataset.initialize()
    dataset.plotImageGrid(dataset.plotImageGridIndexes())


if __name__ == '__main__':
    main()
