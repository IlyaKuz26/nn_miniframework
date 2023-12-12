import numpy as np

class Dense:
  # Layer initialization
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = 0.1 * np.ones((1, n_neurons))

  # Forward pass
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases

  # Backward pass
  def backward(self, dvalues):
    # Gradients of parametrs
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    # Gradients of inputs
    self.dinputs = np.dot(dvalues, self.weights.T)