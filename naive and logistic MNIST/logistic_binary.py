# Step 1: Load the MNIST dataset
import math
from mnist import MNIST
import matplotlib.pyplot as plt

mndata = MNIST('./m')
mndata.gz = True
images, labels = mndata.load_training()

images = [[1 if x > 0 else 0 for x in image] for image in images]
images = images[:1000]
labels = labels[:1000]
X = images

# Step 2: Preprocess the dataset



X_train, X_test, y_train, y_test = X[:680], X[680:], labels[:680], labels[680:]

# Step 3: Define the logistic regression model
def softmax(z):
    # print(1)
    max_vals = []
    for row in z:
        max_val = max(row)
        max_vals.append(max_val)
    
    exps = []
    for i, row in enumerate(z):
        exp_row = []
        for j, val in enumerate(row):
            exp_val = math.exp(val - max_vals[i])
            exp_row.append(exp_val)
        exps.append(exp_row)
    
    sum_exps = []
    for row in exps:
        sum_exp = sum(row)
        sum_exps.append(sum_exp)
    
    softmax_output = []
    for i, row in enumerate(exps):
        softmax_row = []
        for j, val in enumerate(row):
            softmax_val = val / sum_exps[i]
            softmax_row.append(softmax_val)
        softmax_output.append(softmax_row)
    
    return softmax_output

def predict(X, W, b):
    # print(2)
    z = []
    for i, x in enumerate(X):
        z_row = []
        for j, w in enumerate(W):
            dot_product = sum([x[k] * w[k] for k in range(len(x))])
            z_row.append(dot_product + b[0][j])
        z.append(z_row)
    return softmax(z)

# Step 4: Define the loss function
def cross_entropy_loss(y, y_hat):
    # print(3)
    loss = 0
    for i, row in enumerate(y):
        for j, val in enumerate(row):
            if val == 1:
                loss -= math.log(y_hat[i][j])
    return loss

# Step 5: Train the model using gradient descent
learning_rates =  [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
num_epochs = 100
batch_size = 128
num_batches = len(X_train) // batch_size

W = [[0 for _ in range(len(X_train[0]))] for _ in range(10)]
b = [[0 for _ in range(10)]]

loss = 0
accuracies = []

for learning_rate in learning_rates:
    accuracy = float('-inf')
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            # print(9)
            X_batch = X_train[batch*batch_size : (batch+1)*batch_size]
            y_batch = [[1 if i == label else 0 for i in range(10)] for label in y_train[batch*batch_size : (batch+1)*batch_size]]
            y_hat = predict(X_batch, W, b)
            loss = cross_entropy_loss(y_batch, y_hat)
            
            dW = [[0 for _ in range(len(X_train[0]))] for _ in range(10)]
            db = [[0 for _ in range(10)]]
            
            for i, x in enumerate(X_batch):
                for j in range(len(W)):
                    for k in range(len(x)):
                        dW[j][k] += x[k] * (y_hat[i][j] - y_batch[i][j])
                    db[0][j] += y_hat[i][j] - y_batch[i][j]
            
            dW = [[d / batch_size for d in row] for row in dW]
            db = [[d / batch_size for d in row] for row in db]
            
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j] -= learning_rate * dW[i][j]
            
            for i in range(len(b)):
                for j in range(len(b[0])):
                    b[i][j] -= learning_rate * db[i][j]
        
        y_pred = []
        for x in X_test:
            y_pred.append(predict([x], W, b)[0])
        
        y_pred_labels = [max(enumerate(pred), key=lambda x: x[1])[0] for pred in y_pred]
        accuracy = max(accuracy,  sum([1 if y_pred_labels[i] == y_test[i] else 0 for i in range(len(y_pred_labels))]) / len(y_pred_labels))
    accuracies.append(accuracy)
    print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f} %'.format(epoch+1, loss, accuracy*100))


plt.plot(learning_rates, accuracies, label="Pixel Values")
plt.xlabel("Learning rates")
plt.ylabel("Accuracy")
plt.title("Learning rate vs Accuracy for The MNIST Digit Dataset - Binary")
plt.legend()
plt.show()