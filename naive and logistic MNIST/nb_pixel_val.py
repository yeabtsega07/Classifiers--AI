import math
import matplotlib.pyplot as plt
from mnist import MNIST


class NaiveBayesMNIST:
    def __init__(self, laplace=2):
        self.laplace = laplace
        self.freq_distribution = []
        self.tot_freq = []

    def train(self, images, labels):
        pixels = len(images[0])

        self.freq_distribution = [[self.laplace for _ in range(10)] for _ in range(pixels)]
        self.tot_freq = [self.laplace * pixels * 255 for _ in range(10)]

        for ind in range(len(images)):
            image = images[ind]

            for px in range(pixels):
                if image[px]:
                    self.freq_distribution[px][labels[ind]] += image[px]

            self.tot_freq[labels[ind]] += 255

        for ind in range(10):
            for px in range(pixels):
                self.freq_distribution[px][ind] /= self.tot_freq[ind]

    def predict(self, image):
        probs = [0 for _ in range(10)]
        pixels = len(image)

        for px in range(pixels):
            if image[px] > 160:
                for num in range(10):
                    probs[num] += math.log(self.freq_distribution[px][num])
            else:
                for num in range(10):
                    probs[num] += math.log(1 - self.freq_distribution[px][num])

        return probs.index(max(probs))

    def test(self, images, labels):
        tests = 100
        correct = 0

        for test in range(tests):
            predicted = self.predict(images[test])
            actual = labels[test]

            if predicted == actual:
                correct += 1

        accuracy = correct / tests
        return accuracy


def main():
    mndata = MNIST('./m')
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    laplaces = [0.1, 0.5, 1, 2, 10, 50, 100, 500]
    results = []

    for lp in laplaces:
        nb = NaiveBayesMNIST(laplace=lp)
        nb.train(train_images, train_labels)
        accuracy = nb.test(test_images, test_labels)
        results.append(accuracy)
        print("Accuracy - pixel val (Laplace", lp, "):", accuracy * 100, "%")

    plt.plot(laplaces, results)
    plt.xlabel("Laplace Smoothing")
    plt.ylabel("Accuracy")
    plt.title("Laplace Smoothing vs Accuracy for the MNIST Digit Dataset")
    plt.show()


if __name__ == "__main__":
    main()
