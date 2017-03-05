import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Model(object):
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(m, n)
            for n, m in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(m, 1) for m in layers[1:]]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def fit(self, training_data, n_epochs, batch_size, learning_rate,
            test_data):
        for j in range(n_epochs):
            np.random.shuffle(training_data)
            minibatches = [training_data[k:k+batch_size]
                           for k in range(0, len(training_data), batch_size)]
            for minibatch in minibatches:
                self.update(minibatch, learning_rate)
            print('epoch {}/{}: train_acc {:.4f}, test_acc {:.4f}'.format(
                    j+1, n_epochs, self.evaluate(training_data),
                    self.evaluate(test_data)))

    def update(self, minibatch, learning_rate):
        d_weights = [np.zeros(w.shape) for w in self.weights]
        d_biases = [np.zeros(b.shape) for b in self.biases]

        for x, y in minibatch:
            dC_dws, dC_dbs = self.backprop(x, y)
            d_weights = [dw+ddw for dw, ddw in zip(d_weights, dC_dws)]
            d_biases = [db+ddb for db, ddb in zip(d_biases, dC_dbs)]

        self.weights = [w-(learning_rate/len(minibatch))*dw
                        for w, dw in zip(self.weights, d_weights)]
        self.biases = [b-(learning_rate/len(minibatch))*db
                       for b, db in zip(self.biases, d_biases)]

    def backprop(self, x, y):
        dC_dws = [np.zeros(w.shape) for w in self.weights]
        dC_dbs = [np.zeros(b.shape) for b in self.biases]
        # set first activation layer as input x
        a = x
        A = [x]
        Z = []
        # feedforward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a)+b
            Z.append(z)
            a = sigmoid(z)
            A.append(a)
        # compute error vector for output layer L
        L = len(self.layers)
        error = (A[L-1]-y)*sigmoid_prime(Z[L-2])
        dC_dws[L-2] = np.dot(error, A[L-2].transpose())
        dC_dbs[L-2] = error
        # backpropagate
        for i in range(L-3, -1, -1):
            error = np.dot(self.weights[i+1].transpose(), error)*sigmoid_prime(Z[i])
            dC_dws[i] = np.dot(error, A[i].transpose())
            dC_dbs[i] = error
        # output gradient
        return dC_dws, dC_dbs

    def evaluate(self, data):
        return sum(int(np.argmax(self.feedforward(x)) == np.argmax(y))
                   for x, y in data)/len(data)
