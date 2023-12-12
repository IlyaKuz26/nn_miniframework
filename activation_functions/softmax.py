import numpy as np


class Softmax:
  # Forward pass
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.output = probabilities

  # Backward pass
  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues)
    # Enumerate outputs and gradients
    for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
      single_output = single_output.reshape(-1, 1)
      # Calculate Jacobian matrix of the output
      jacobian_matrix = np.diagflat(single_output) - \
                        np.dot(single_output, single_output.T)
      # Calculate sample-wise gradient
      self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)