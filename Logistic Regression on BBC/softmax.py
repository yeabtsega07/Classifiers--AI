import math
import bbcDataset

X_train = bbcDataset.X_train
X_test = bbcDataset.X_test
y_train = bbcDataset.y_train
y_test = bbcDataset.y_test
num_class = 5

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
learning_rates = [0.001]
num_epochs = 100
batch_size = 128
num_batches = len(X_train) // batch_size

W = [[0 for _ in range(len(X_train[0]))] for _ in range(num_class)]
b = [[0 for _ in range(num_class)]]

loss = 0

for learning_rate in learning_rates:
    max_accuracy = 0
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            # print(9)
            X_batch = X_train[batch*batch_size : (batch+1)*batch_size]
            y_batch = [[1 if i == label else 0 for i in range(num_class)] for label in y_train[batch*batch_size : (batch+1)*batch_size]]
            y_hat = predict(X_batch, W, b)
            loss = cross_entropy_loss(y_batch, y_hat)
            
            dW = [[0 for _ in range(len(X_train[0]))] for _ in range(num_class)]
            db = [[0 for _ in range(num_class)]]
            
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
        accuracy = sum([1 if y_pred_labels[i] == y_test[i] else 0 for i in range(len(y_pred_labels))]) / len(y_pred_labels)
        max_accuracy = max(accuracy,max_accuracy)
        print(accuracy)
    print(f'Learing Rate: {learning_rate}, Accuracy: {max_accuracy}')