import numpy as np


class ReLU:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor >= 0)