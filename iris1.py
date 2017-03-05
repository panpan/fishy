from sklearn import datasets, preprocessing
import numpy as np
import network1

iris = datasets.load_iris()
X, y = iris.data, iris.target

lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(y)

data = [(x.reshape((-1,1)), y.reshape((-1,1))) for x, y in zip(X, Y)]

np.random.shuffle(data)

train_test_split = 0.7
split_idx = int(train_test_split*len(data))
training_data, test_data = data[:split_idx], data[split_idx:]

model = network1.Model([4, 8, 3])
model.fit(training_data, 200, 4, 0.1, test_data)
