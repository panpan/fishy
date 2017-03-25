import gzip
import pickle
from sklearn import preprocessing
import neural_net

with gzip.open('mnist.pkl.gz', 'r') as f:
    training_set, validation_set, test_set = pickle.load(
        f, encoding='latin1')

X_train, y_train = training_set[0], training_set[1]
X_test, y_test = test_set[0], test_set[1]

lb = preprocessing.LabelBinarizer()
Y_train, Y_test = lb.fit_transform(y_train), lb.fit_transform(y_test)

training_data = [(x.reshape((-1, 1)), y.reshape((-1, 1)))
                 for x, y in zip(X_train, Y_train)]
test_data = [(x.reshape((-1, 1)), y.reshape((-1, 1)))
             for x, y in zip(X_test, Y_test)]

if __name__ == '__main__':
    model = neural_net.Model([784, 50, 10])
    model.fit(training_data, 50, 10, 3.0, test_data)
