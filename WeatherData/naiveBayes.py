from csv import reader
from math import exp


import matplotlib.pyplot as plt
import numpy as np


class NaiveBayes:
    def __init__(self, laplace_smoothing=1.0):
        self.prior = {}
        self.conditional_prob = {}
        self.classes = []
        self.laplace_smoothing = laplace_smoothing

    def train(self, features, labels):
        num_examples = len(features)
        num_features = len(features[0])
        self.classes = set(labels)

        for label in self.classes:
            class_examples = [features[i]
                              for i in range(num_examples) if labels[i] == label]
            self.prior[label] = len(class_examples) / num_examples

            feature_prob = {}
            for feature_idx in range(num_features):
                feature_values = [example[feature_idx]
                                  for example in class_examples]
                value_counts = {value: feature_values.count(
                    value) for value in set(feature_values)}
                feature_prob[feature_idx] = {
                    value: (count + self.laplace_smoothing) / (len(class_examples) + self.laplace_smoothing * len(value_counts)) for value, count in value_counts.items()}

            self.conditional_prob[label] = feature_prob

    def predict(self, features):
        predictions = []
        for feature in features:
            class_probs = {label: self.prior[label] for label in self.classes}
            for label, class_cond_prob in self.conditional_prob.items():
                for feature_idx, feature_value in enumerate(feature):
                    if feature_value in class_cond_prob.get(feature_idx, {}):
                        class_probs[label] *= class_cond_prob[feature_idx][feature_value]
                    else:
                        # Laplace smoothing for unseen feature values
                        class_probs[label] *= self.laplace_smoothing / \
                            (len(class_cond_prob[feature_idx]
                                 ) + self.laplace_smoothing)

            predicted_label = max(class_probs, key=class_probs.get)
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


train_data = load_csv('./train.csv')[1:]
test_data = load_csv('./test.csv')[1:]

train_features, train_labels = extract_features_labels(train_data)
test_features, test_labels = extract_features_labels(test_data)

train_features_encoded = one_hot_encode(train_features)
test_features_encoded = one_hot_encode(test_features)

model = NaiveBayes()
model.train(train_features_encoded, train_labels)

train_predictions = model.predict(train_features_encoded)
train_accuracy = sum(1 for i in range(len(train_labels))
                     if train_labels[i] == train_predictions[i]) / len(train_labels)

test_predictions = model.predict(test_features_encoded)
test_accuracy = sum(1 for i in range(len(test_labels))
                    if test_labels[i] == test_predictions[i]) / len(test_labels)

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
accuracy_train = []
accuracy_test = []

for learning_rate in learning_rates:
    model = NaiveBayes(learning_rate)
    model.train(train_features_encoded, train_labels)

    train_predictions = model.predict(train_features_encoded)
    train_accuracy = sum(1 for i in range(len(train_labels))
                         if train_labels[i] == train_predictions[i]) / len(train_labels)
    accuracy_train.append(train_accuracy)

    test_predictions = model.predict(test_features_encoded)
    test_accuracy = sum(1 for i in range(len(test_labels))
                        if test_labels[i] == test_predictions[i]) / len(test_labels)
    accuracy_test.append(test_accuracy)

plt.plot(learning_rates, accuracy_train, label='Train Accuracy')
plt.plot(learning_rates, accuracy_test, label='Test Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Naive Bayes Accuracy with Different Learning Rates')
plt.legend()
plt.show()

print("Train Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
