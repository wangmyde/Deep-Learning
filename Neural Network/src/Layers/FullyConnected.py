import numpy as np


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size + 1)
        self.input_tensor_plus_bias = None
        self.delta = 1

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        plus_ones = np.ones((batch_size, 1))
        self.input_tensor_plus_bias = np.hstack((self.input_tensor,plus_ones))
        output_tensor = np.dot(self.input_tensor_plus_bias, self.weights.T)
        return output_tensor

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        error_tensor_lower_trans = np.dot(self.weights.T, error_tensor.T) # Got input_size * batch_size. It's transpose of error_tensor.
        self.weights = self.weights - self.delta * np.dot(self.error_tensor.T, self.input_tensor_plus_bias)
        error_tensor_lower_minus_bias_trans = np.delete(error_tensor_lower_trans, -1, 0)
        error_tensor = error_tensor_lower_minus_bias_trans.T
        return error_tensor


    def get_gradient_weights(self):
        gradient_weights = np.dot(self.error_tensor.T, self.input_tensor_plus_bias)
        return gradient_weights
