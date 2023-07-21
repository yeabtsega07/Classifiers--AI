from collections import Counter
from mnist import MNIST
import matplotlib.pyplot as plt


class NaiveBayeMNIST:
    
    def __init__(self, smooth=0.01):
        self.smooth = smooth

    def get_priors(self, y):
        priors = Counter(y)
        for i in range(10):
            priors[i]
        return priors

    def train_nb(self, train_images, train_labels):
        edge_counts = [[self.smooth] * 10 for _ in range(28 * 28)]
        
        for i in range(len(train_images)):
            cur_image = train_images[i]
            cur_label = train_labels[i]
            
            for j in range(1, len(cur_image) - 1):
                pixel = cur_image[j]
                prev_pixel = cur_image[j - 1]
                next_pixel = cur_image[j + 1]
                
                if pixel > prev_pixel and pixel > next_pixel:
                    edge_counts[j][cur_label] += 1

        return edge_counts

    def predict_nb(self, images, labels):
        edge_counts = self.train_nb(images, labels)
        priors = self.get_priors(labels)
        prediction = []
        
        for pixel in range(len(edge_counts)):
            pixel_counts = edge_counts[pixel]
            pixel_probs = []
            
            for i, count in enumerate(pixel_counts):
                cur_prob = count / (priors[i] + 2 * self.smooth)
                pixel_probs.append(cur_prob)
                
            prediction.append(pixel_probs)
        
        return prediction

    def test(self, test_images, test_labels, prediction):
        valid = 0
        
        for i, image in enumerate(test_images):
            label = test_labels[i]
            predictions = [1] * 10
            
            for j, pixel in enumerate(image):
                if j > 0 and j < len(image) - 1:
                    prev_pixel = image[j - 1]
                    next_pixel = image[j + 1]
                    
                    if pixel > prev_pixel and pixel > next_pixel:
                        pixel_probs = prediction[j]
                        
                        for k in range(10):
                            predictions[k] *= pixel_probs[k]
                    
            predicted = 0
            max_ = predictions[0]
            
            for i in range(10):
                if predictions[i] > max_:
                    max_ = predictions[i]
                    predicted = i
            
            if predicted == label:
                valid += 1
        
        return valid / len(test_images)



def main():

    # Load MNIST dataset
    mndata = MNIST('./m')
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    smooth_values = [0.1, 0.5, 1, 2, 10, 50, 100, 500]
    accuracy_values = []

    for smooth in smooth_values:
        nb = NaiveBayeMNIST(smooth)
        prediction = nb.predict_nb(test_images, test_labels)
        accuracy = nb.test(test_images, test_labels, prediction) * 100
        accuracy_values.append(accuracy)
        print("Accuracy (Smooth =", smooth, "):", accuracy, "%")

    plt.plot(smooth_values, accuracy_values)
    plt.xlabel("Smooth Value")
    plt.ylabel("Accuracy")
    plt.title("Smooth Value vs Accuracy for the MNIST Digit Dataset")
    plt.show()

if __name__ == "__main__":
    main()