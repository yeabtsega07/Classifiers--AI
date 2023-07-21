import math
import matplotlib.pyplot as plt

# main
from collections import Counter
from mnist import MNIST

# Load MNIST dataset
mndata = MNIST('./m')
mndata.gz = True
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

class NaiveBayeMNIST:
    def __init__(self, smooth=0.01):
        self.smooth = smooth

    def get_priors(self, y):
        priors = Counter(y)
        for i in range(10):
            priors[i]
        return priors

    def train_binary(self, train_images, train_labels):
        on_pixels = [[self.smooth for i in range(10)] for _ in range(28 * 28)]

        for i in range(len(train_images)):
            cur_image = train_images[i]
            cur_label = train_labels[i]
            for j in range(len(cur_image)):
                if cur_image[j] > 0:
                    on_pixels[j][cur_label] += 1

        return on_pixels

    def predict_binary(self, images, labels):
        on_pixels = self.train_binary(images, labels)
        priors = self.get_priors(labels)
        prediction = []

        for pixel in on_pixels:
            probabilities = []

            for i, count in enumerate(pixel):
                cur_prob = count / (priors[i] + 10 * self.smooth)
                probabilities.append(cur_prob)

            prediction.append(probabilities)

        return prediction

    def test(self, test_images, test_labels, prediction):
        valid = 0
        for i, image in enumerate(test_images):
            label = test_labels[i]
            predictions = [1] * 10

            for j, pixel in enumerate(image):
                if pixel > 0:
                    for k in range(10):
                        predictions[k] *= prediction[j][k]
                else:
                    for k in range(10):
                        predictions[k] *= (1 - prediction[j][k])

            predicted = 0
            max_ = predictions[0]
            for i in range(10):
                if predictions[i] > max_:
                    max_ = predictions[i]
                    predicted = i
            if predicted == label:
                valid += 1

        return valid / len(test_images)


laplaces = [0.1, 0.5, 1, 2, 10, 50, 100, 500]
results_binary = []

for lp in laplaces:
    nb = NaiveBayeMNIST(smooth=lp)
    prediction = nb.predict_binary(test_images, test_labels)
    accuracy = nb.test(test_images, test_labels, prediction) * 100
    results_binary.append(accuracy)
    print("Accuracy - Binary (Laplace", lp, "):", accuracy, "%")

plt.plot(laplaces, results_binary, label="Binary")
plt.xlabel("Laplace Smoothing")
plt.ylabel("Accuracy")
 
  
