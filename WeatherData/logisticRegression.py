import matplotlib.pyplot as plt
from csv import reader
from math import exp
import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def train(self, features, labels):
        num_features = len(features[0])
        num_examples = len(features)
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            for i in range(num_examples):
                y = labels[i]
                z = self.bias + np.dot(self.weights, features[i])
                sigmoid = 1 / (1 + np.exp(-z))
                error = sigmoid - y

                self.bias -= self.learning_rate * error
                self.weights -= self.learning_rate * \
                    error * np.array(features[i])

    def predict(self, features):
        predictions = []
        for i in range(len(features)):
            z = self.bias + np.dot(self.weights, features[i])
            sigmoid = 1 / (1 + np.exp(-z))
            predictions.append(1 if sigmoid >= 0.5 else 0)
        return predictions


class MulticlassLogisticRegression:
    def __init__(self, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.models = {}

    def train(self, features, labels):
        unique_labels = list(set(labels))
        for label in unique_labels:
            binary_labels = [1 if l == label else 0 for l in labels]
            model = LogisticRegression(self.learning_rate, self.num_epochs)
            model.train(features, binary_labels)
            self.models[label] = model

    def predict(self, features):
        predictions = []
        for feature in features:
            label_predictions = {}
            for label, model in self.models.items():
                label_predictions[label] = model.predict([feature])[0]
            predicted_label = max(label_predictions, key=label_predictions.get)
            predictions.append(predicted_label)
        return predictions


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if row:
                dataset.append(row)
    return dataset


def extract_features_labels(data):
    features = []
    labels = []
    for row in data:
        features.append(row[:-1])
        labels.append(row[-1])
    return features, labels


def one_hot_encode(features):
    all_categories = set()
    for row in features:
        for val in row:
            all_categories.add(val)

    all_categories = sorted(list(all_categories))
    encoded_features = []
    for row in features:
        encoded_row = [0] * len(all_categories)
        for i, category in enumerate(all_categories):
            if category in row:
                encoded_row[i] = 1
        encoded_features.append(encoded_row)

    return encoded_features


def encode_labels(labels):
    encoded_labels = []
    for label in labels:
        if label == 'Normal':
            encoded_labels.append(0)
        else:
            encoded_labels.append(1)
    return encoded_labels


train_data = load_csv('./train.csv')[1:]
test_data = load_csv('./test.csv')[1:]

train_features, train_labels = extract_features_labels(train_data)
test_features, test_labels = extract_features_labels(test_data)

train_features_encoded = one_hot_encode(train_features)
test_features_encoded = one_hot_encode(test_features)

train_labels_encoded = encode_labels(train_labels)
test_labels_encoded = encode_labels(test_labels)

model = MulticlassLogisticRegression(learning_rate=0.02, num_epochs=100)
model.train(train_features_encoded, train_labels_encoded)

train_predictions = model.predict(train_features_encoded)
train_accuracy = sum(1 for i in range(len(train_labels_encoded))
                     if train_labels_encoded[i] == train_predictions[i]) / len(train_labels_encoded)

test_predictions = model.predict(test_features_encoded)
test_accuracy = sum(1 for i in range(len(test_labels_encoded))
                    if test_labels_encoded[i] == test_predictions[i]) / len(test_labels_encoded)

print("Train Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))


learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
laplace_smoothing = [0.1, 0.5, 1.0, 10, 100]
train_accuracies = []
test_accuracies = []

for lr in learning_rates:
    for ls in laplace_smoothing:
        model = MulticlassLogisticRegression(learning_rate=lr, num_epochs=100)
        model.train(train_features_encoded, train_labels_encoded)

        train_predictions = model.predict(train_features_encoded)
        train_accuracy = sum(1 for i in range(len(train_labels_encoded))
                             if train_labels_encoded[i] == train_predictions[i]) / len(train_labels_encoded)

        test_predictions = model.predict(test_features_encoded)
        test_accuracy = sum(1 for i in range(len(test_labels_encoded))
                            if test_labels_encoded[i] == test_predictions[i]) / len(test_labels_encoded)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print("Learning Rate: {}, Laplace Smoothing: {}".format(lr, ls))
        print("Train Accuracy: {:.2f}%".format(train_accuracy * 100))
        print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
        print()

# Plotting the accuracy change based on the hyperparameter variations
plt.figure(figsize=(10, 6))
plt.plot(range(len(train_accuracies)),
         train_accuracies, label='Train Accuracy')
plt.plot(range(len(test_accuracies)), test_accuracies, label='Test Accuracy')
plt.xticks(range(len(train_accuracies)), [f"LR={lr}, LS={ls}" for lr in learning_rates for ls in laplace_smoothing],
           rotation=45)
plt.xlabel('Hyperparameters')
plt.ylabel('Accuracy')
plt.title('Accuracy Change based on Hyperparameter Variations')
plt.legend()
plt.tight_layout()
plt.show()
