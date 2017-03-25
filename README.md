# Multilayer neural network modeled after Michael Nielsen

## iris

Fisher's [Iris data set][iris_data] consists of measurements collected from 150 samples of iris flowers. Fifty samples were taken from each of three species (Iris setosa, Iris versicolor, Iris virginica) and four measurements (in cm) were taken for each sample: 1. sepal length, 2. sepal width, 3. petal length, 4. petal width.

Plotting pairwise relationships in the data shows that one species (Iris setosa) is linearly separable, while the other two are not:

![iris_plot]

[iris][iris] uses a [neural network][neural_network] based on Michael Nielsen's to classify the data set with 98-100% accuracy. The model is a three-layer neural network with one hidden layer, using sigmoid neurons with a quadratic cost function. The data is split into a training set and a test set, and the model is fitted to the training set via mini-batch stochastic gradient descent. Accuracy values are reported for both sets at the end of each epoch.

## mnist

The [MNIST database][mnist_data] of handwritten digits (originally from [here][mnist_lecun]) consists of 28x28 images of handwritten digits along with their labels 0-9. The data is split into a 50,000-image training set and a 10,000-image test set. [mnist][mnist] uses another three-layer neural network with a 50-neuron hidden layer to classify the test set with 96% accuracy.

For more information on neural networks and implementation, see Michael Nielsen's [Neural Networks and Deep Learning][mnielsen].

[iris_data]: https://archive.ics.uci.edu/ml/datasets/Iris
[iris_plot]: ./iris_plot.png
[iris]: ./iris.ipynb
[neural_network]: ./neural_network.py
[mnist]: ./mnist.py
[mnist_data]: http://deeplearning.net/data/mnist/
[mnist_lecun]: http://yann.lecun.com/exdb/mnist/
[mnielsen]: http://neuralnetworksanddeeplearning.com/
