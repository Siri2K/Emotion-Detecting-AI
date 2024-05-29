from data import EmotionImages


def main():
    dataset: EmotionImages = EmotionImages()
    dataset.initialize()
    dataset.plotImageGrid(dataset.plotImageGridIndexes())
    dataset.pixelIntensityDistributionClass()
    dataset.classDistribution()
    dataset.display()

if __name__ == '__main__':
    main()
