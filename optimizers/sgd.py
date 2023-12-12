class SGD:
  # Initialize optimizerr
  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate

  # Update parameters
  def update_params(self, layer):
    layer.weights += -self.learning_rate * layer.dweights
    layer.biases += -self.learning_rate * layer.dbiases