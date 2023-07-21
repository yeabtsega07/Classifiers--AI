from collections import defaultdict
import random

doc_class = {}

filename = 'bbc\\bbc.classes'
with open(filename, 'r') as file:
    for ind, line in enumerate(file):
        if ind > 3:
            doc, _class = map(int, line.split())
            doc_class[doc + 1] = _class

matrix = defaultdict(dict)
filename = 'bbc\\bbc.mtx'

unique_words = defaultdict(int)

with open(filename, 'r') as file:
    for ind, line in enumerate(file):
        line = line.strip()
        if ind > 1 and line:
            word, doc, freq = line.split()
            matrix[int(doc)][int(word)] = float(freq)
            unique_words[word] += 1

datasets = []
for doc, words in matrix.items():
    dataset = [0] * 9636
    for word, freq in words.items():
        dataset[word - 1] = freq

    dataset[-1] = doc_class[doc]
    datasets.append(dataset)

random.shuffle(datasets)
splitIdx = int(len(datasets) * 0.7)
training = datasets[:splitIdx]
testing = datasets[splitIdx:]

X_train = [x[:-1] for x in training]
y_train = [x[-1] for x in training]
X_test = [x[:-1] for x in testing]
y_test = [x[-1] for x in testing]

# Export the variables ass Python module
import sys

module_name = "bbcDataset"
module = sys.modules[__name__]

module.X_train = X_train
module.y_train = y_train
module.X_test = X_test
module.y_test = y_test
