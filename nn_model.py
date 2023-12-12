import numpy as np

from layers.dense import Dense
from loss_functions.mse import MSE
from optimizers.sgd import SGD
from utils import divide_into_batches


class NN_Model:
  def __init__(self):
    self.layers = []

  def add(self, layer):
    self.layers.append(layer)

  # Forward pass
  def predict(self, X):
    for layer in self.layers:
      if self.layers.index(layer) == 0:
        layer.forward(X)
        layer_output = layer.output
      else:
        layer.forward(layer_output)
        layer_output = layer.output

    return layer_output

  # Training of neural network
  def train(self, X_train, y_train, epochs=10, batch_size=1,
            loss=MSE, optimizer=SGD, learning_rate=0.01):

    loss_function = loss()
    optimizer = optimizer(learning_rate)

    for epoch in range(epochs):
      X_batches = divide_into_batches(X_train, batch_size)
      y_batches = divide_into_batches(y_train, batch_size)

      for i, (X, y) in enumerate(zip(X_batches, y_batches)):
        forward_output = self.predict(X)
        loss = loss_function.forward(forward_output, y)

        # Accuracy calculation
        if epoch % 10 == 0 and i == 0:
          y_measurement = y.copy()
          predictions = np.argmax(forward_output, axis=1)
          if len(y_measurement.shape) == 2:
            y_measurement = np.argmax(y_measurement, axis=1)
          train_accuracy = np.mean(predictions==y_measurement)

          print(f'epoch: {epoch}, loss {np.mean(loss):.3f}, train_accuracy {train_accuracy:.3f}')

        # Backpropogation
        loss_function.backward(forward_output, y)
        dinputs = loss_function.dinputs
        
        for layer in reversed(self.layers):
          layer.backward(dinputs)
          dinputs = layer.dinputs

        # Optimization of parameters
        for layer in self.layers:
          if isinstance(layer, Dense):
            optimizer.update_params(layer)

  # Testing of neural network
  def test(self, X, y):
    predictions = np.argmax(self.predict(X), axis=1)
    if len(y.shape) == 2:
      y = np.argmax(y, axis=1)

    test_accuracy = np.mean(predictions==y)

    print(f"Test accuracy: {test_accuracy}")