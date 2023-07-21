from collections import defaultdict
import random
from decimal import Decimal, getcontext
from matplotlib import pyplot as plt


class Helper:
    def __init__(self):
        pass

    def extract_raw_data(self):
        raw_data = []
        with open("dataset/bbc.classes") as file:
            for line_num, line in enumerate(file):
                if line_num > 3:
                    document_id, class_id = line.rstrip().split()
                    raw_data.append([document_id, class_id])
        return raw_data

    def extract_terms_raw_data(self):
        terms_raw_data = []
        with open("dataset/bbc.mtx", newline="") as mtx:
            for line_num, line in enumerate(mtx):
                if line_num >= 2:
                    term_id, document_id, freq = line.rstrip().split()
                    terms_raw_data.append([term_id, document_id, freq])

        return terms_raw_data

    def get_classes_prob(self, raw_data):
        classes_count = defaultdict(int)
        total_data = len(raw_data)

        for document_id, class_id in raw_data:
            classes_count[class_id] += 1

        return {
            class_id: Decimal(class_count) / Decimal(total_data)
            for class_id, class_count in classes_count.items()
        }

    def beautify_data(self, raw_data):
        beautified_data = {}
        for document_id, class_id in raw_data:
            beautified_data[document_id] = class_id

        return beautified_data

    def get_terms_probability(self, class_terms):
        class_probabilities = {}
        total_terms = {}
        for class_id, term_freqs in class_terms.items():
            total_freq = sum(term_freqs.values())
            total_terms[class_id] = total_freq
            term_probabilities = {
                term_id: Decimal(freq) / Decimal(total_freq)
                for term_id, freq in term_freqs.items()
            }
            class_probabilities[class_id] = term_probabilities

        return class_probabilities, total_terms

    def get_document_terms(self):
        document_terms = {}

        with open("dataset/bbc.mtx", newline="") as mtx:
            for line_num, line in enumerate(mtx):
                if line_num >= 2:
                    term_id, document_id, _ = (val for val in line.split())
                    document_dict = document_terms.setdefault(
                        str(int(document_id) - 1), []
                    )
                    document_dict.append(term_id)

        return document_terms


class NaiveBayes:
    def __init__(
        self,
        classes_prob,
        terms_prob,
        document_terms,
        total_terms,
    ):
        self.classes_prob = classes_prob
        self.terms_prob = terms_prob
        self.document_terms = document_terms
        self.total_terms_dict = total_terms

    def get_dic_classification(self, data, laplace_smoothing):
        probabilities = {}
        for document_id in data:
            probabilities[document_id] = self.classifier(document_id, laplace_smoothing)

        return probabilities

    def classifier(self, document_id, laplace_smoothing):
        max_probability = float("-inf")
        max_class_label = None

        for class_label in self.classes_prob:
            total_terms = self.total_terms_dict[class_label]
            probability = 1.0
            for term in self.document_terms[document_id]:
                if term in self.terms_prob[class_label]:
                    probability = Decimal(probability) * Decimal(
                        self.terms_prob[class_label][term]
                    )
                else:
                    probability = Decimal(probability) * (
                        Decimal(laplace_smoothing)
                        / Decimal(total_terms + laplace_smoothing)
                    )

            probability = Decimal(probability) * Decimal(self.classes_prob[class_label])
            if probability > max_probability:
                max_probability = probability
                max_class_label = class_label

        return max_class_label


class Runner:
    def __init__(self):
        self.utils = Helper()
        self.training_data, self.test_data = self.split_dataset()

        self.prior_prob = self.utils.get_classes_prob(self.training_data)
        self.training_documents_data = self.utils.beautify_data(self.training_data)

        self.test_documents_data = self.utils.beautify_data(self.test_data)

        self.training_terms_data = self.split_terms_dataset(
            self.training_documents_data
        )
        self.terms_prior_prob, self.total_terms = self.utils.get_terms_probability(
            self.training_terms_data
        )

        self.document_terms = self.utils.get_document_terms()

    def do_naive_bayes(self, laplace_smoothing):
        naive_bayes = NaiveBayes(
            self.prior_prob,
            self.terms_prior_prob,
            self.document_terms,
            self.total_terms,
        )
        predicted_labels = naive_bayes.get_dic_classification(
            self.test_documents_data, laplace_smoothing
        )

        return self.calculate_accuracy(predicted_labels, self.test_documents_data)

    def split_dataset(self, test_data_ratio=0.2):
        raw_data = self.utils.extract_raw_data()

        split_index = int(round(test_data_ratio * len(raw_data)))

        random.shuffle(raw_data)

        test_data = raw_data[:split_index]
        training_data = raw_data[split_index:]

        return training_data, test_data

    def split_terms_dataset(self, training_documents_data):
        raw_terms_data = self.utils.extract_terms_raw_data()
        class_terms = defaultdict(lambda: defaultdict(int))

        for term_id, document_id, freq in raw_terms_data:
            if document_id in training_documents_data:
                class_id = training_documents_data[document_id]
                class_terms[class_id][term_id] += float(freq)

        return class_terms

    def calculate_accuracy(
        self, predicted_labels, original_labels
    ):
        correct_count = 0
        total_count = len(predicted_labels)

        for document_id, predicted_label in predicted_labels.items():
            original_label = original_labels[document_id]
            if predicted_label == original_label:
                correct_count += 1

        accuracy = correct_count / total_count
        return accuracy * 100


def main():
    run = Runner()
    accuracies = []
    laplace = [0.1, 0.5, 1.0, 10, 100, 300, 500]
    for i in laplace:
        accuracy = run.do_naive_bayes(i)
        accuracies.append((accuracy, i))

    smoothing_values = [entry[1] for entry in accuracies]
    accuracy_values = [entry[0] for entry in accuracies]

    plt.plot(smoothing_values, accuracy_values, marker="o")
    plt.xlabel("Smoothing")
    plt.ylabel("Accuracy (%)")
    plt.title("Smoothing vs Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
