from data import EmotionImages, display


def main():
    dataset: EmotionImages = EmotionImages()
    dataset.initialize()
    dataset.plotImageGrid(dataset.gatherImageIndexes())
    dataset.plotPixelIntensityForSample()
    dataset.pixelIntensityDistributionClass()
    dataset.classDistribution()
    display()


if __name__ == '__main__':
    main()
