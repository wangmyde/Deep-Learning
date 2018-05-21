import numpy as np
from sklearn.datasets import load_iris
from random import shuffle


def gradient_check(layers, input_tensor, label_tensor):
    epsilon = 1e-5
    difference = np.zeros_like(input_tensor)
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            plus_epsilon = input_tensor.copy()
            plus_epsilon[i, j] += epsilon
            minus_epsilon = input_tensor.copy()
            minus_epsilon[i, j] -= epsilon

            activation_tensor = input_tensor.copy()
            for layer in layers[:-1]:
                activation_tensor = layer.forward(activation_tensor)
            layers[-1].forward(activation_tensor, label_tensor)

            error_tensor = layers[-1].backward(label_tensor)
            for layer in reversed(layers[:-1]):
                error_tensor = layer.backward(error_tensor)
            analytical_derivative = error_tensor[i, j]

            for layer in layers[:-1]:
                plus_epsilon = layer.forward(plus_epsilon)
                minus_epsilon = layer.forward(minus_epsilon)
            upper_error = layers[-1].forward(plus_epsilon, label_tensor)
            lower_error = layers[-1].forward(minus_epsilon, label_tensor)

            numerical_derivative = (upper_error - lower_error) / (2 * epsilon)

            normalizing_constant = max(np.abs(analytical_derivative), np.abs(numerical_derivative))

            if normalizing_constant < 1e-15:
                difference[i, j] = 0
            else:
                difference[i, j] = np.abs(analytical_derivative - numerical_derivative) / normalizing_constant
    return difference


def gradient_check_weights(layers, input_tensor, label_tensor, bias):
    epsilon = 1e-5
    if bias:
        weights = layers[0].bias
    else:
        weights = layers[0].weights
    difference = np.zeros_like(weights)

    it = np.nditer(weights, flags=['multi_index'])
    while not it.finished:
        plus_epsilon = weights.copy()
        plus_epsilon[it.multi_index] += epsilon
        minus_epsilon = weights.copy()
        minus_epsilon[it.multi_index] -= epsilon

        activation_tensor = input_tensor.copy()
        if bias:
            layers[0].bias = weights
        else:
            layers[0].weights = weights
        for layer in layers[:-1]:
            activation_tensor = layer.forward(activation_tensor)
        layers[-1].forward(activation_tensor, label_tensor)

        error_tensor = layers[-1].backward(label_tensor)
        for layer in reversed(layers[:-1]):
            error_tensor = layer.backward(error_tensor)
        if bias:
            analytical_derivative = layers[0].get_gradient_bias()
        else:
            analytical_derivative = layers[0].get_gradient_weights()

        analytical_derivative = analytical_derivative[it.multi_index]

        if bias:
            layers[0].bias = plus_epsilon
        else:
            layers[0].weights = plus_epsilon
        plus_epsilon_activation = input_tensor.copy()
        for layer in layers[:-1]:
            plus_epsilon_activation = layer.forward(plus_epsilon_activation)

        if bias:
            layers[0].bias = minus_epsilon
        else:
            layers[0].weights = minus_epsilon
        minus_epsilon_activation = input_tensor.copy()
        for layer in layers[:-1]:
            minus_epsilon_activation = layer.forward(minus_epsilon_activation)

        upper_error = layers[-1].forward(plus_epsilon_activation, label_tensor)
        lower_error = layers[-1].forward(minus_epsilon_activation, label_tensor)

        numerical_derivative = (upper_error - lower_error) / (2 * epsilon)

        normalizing_constant = max(np.abs(analytical_derivative), np.abs(numerical_derivative))

        if normalizing_constant < 1e-15:
            difference[it.multi_index] = 0
        else:
            difference[it.multi_index] = np.abs(analytical_derivative - numerical_derivative) / normalizing_constant

        it.iternext()
    return difference


def shuffle_data(input_tensor, label_tensor):
    index_shuffling = [i for i in range(input_tensor.shape[0])]
    shuffle(index_shuffling)
    shuffled_input = [input_tensor[i, :] for i in index_shuffling]
    shuffled_labels = [label_tensor[i, :] for i in index_shuffling]
    return np.array(shuffled_input), np.array(shuffled_labels)


class RandomData:
    def __init__(self, input_size, batch_size, categories):
        self.input_size = input_size
        self.batch_size = batch_size
        self.categories = categories
        self.label_tensor = np.zeros([self.batch_size, self.categories])

    def forward(self):
        input_tensor = np.random.random([self.batch_size, self.input_size])

        self.label_tensor = np.zeros([self.batch_sizeself.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        return input_tensor, self.label_tensor


class IrisData:
    def __init__(self):
        self.data = load_iris()
        self.data.target = self.data.target.T

        # print(self.data.target.shape)
        self.label_tensor = np.zeros([150, 3])

        for i in range(150):
            self.label_tensor[i, self.data.target[i]] = 1

        self.input_tensor, self.label_tensor = shuffle_data(np.array(self.data.data), self.label_tensor)

    def forward(self):
        return self.input_tensor[0:100, :], self.label_tensor[0:100, :]

    def get_test_set(self):
        return self.input_tensor[100:150, :], self.label_tensor[100:150, :]

