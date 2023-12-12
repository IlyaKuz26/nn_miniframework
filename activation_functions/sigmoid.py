import numpy as np


class Sigmoid:
  # Forward pass
  def forward(self, inputs):
    self.inputs = inputs
    self.output = 1/(1 + np.exp(-inputs))
    # print("sigmoid output", self.output)

  # Backward pass
  def backward(self, dvalues):
    # print('sigmoid backward ', self.output, dvalues)
    self.dinputs = (self.output * (1 - self.output)) * dvalues
    # print('back sigmoid', self.dinputs)