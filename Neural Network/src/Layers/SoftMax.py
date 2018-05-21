import numpy as np
import math

class SoftMax:
    def __init__(self):
        self.X = None

    def forward(self, input_tensor, label_tensor):
        self.X = self.predict(input_tensor)
        loss = np.sum(label_tensor * (-np.log(self.X)))
        return loss

    def backward(self, label_tensor):
        label_correct = np.subtract(self.X, label_tensor)
        return label_correct

    def predict(self, input_tensor):
        input_tensor_change = input_tensor - np.max(input_tensor)
        expo = np.exp(input_tensor_change)
        exp_sum = np.sum(expo, axis = 1)
        final = np.zeros((expo.shape[0],expo.shape[1]))
        for i in range(expo.shape[0]):
            for j in range(expo.shape[1]):
                final[i,j] = expo[i,j] / exp_sum[i]
        return final