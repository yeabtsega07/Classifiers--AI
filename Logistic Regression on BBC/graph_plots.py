import matplotlib.pyplot as plt

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
accuracies = [0.84722, 0.86552, 0.88927, 0.92132, 0.94222, 0.96001]

plt.plot(learning_rates, accuracies, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Learning Rate vs Accuracy - BBC Dataset Logistic Regression')
plt.grid(True)
plt.show()
