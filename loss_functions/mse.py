import numpy as np

from loss import Loss


class MSE(Loss):
  # Forward pass
  def forward(self, y_pred, y_true):
    # Number of samples
    n_labels = y_pred.shape[0]
    # Clip predicted values
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # If labels are discrete, turn them into one-hot vectors
    if len(y_true.shape) == 1:
      y_true_one_hot = np.zeros((n_labels, 10))
      y_true_one_hot[range(n_labels), y_true] = 1
      y_true = y_true_one_hot
    # Calculate loss
    likehoods = np.square((y_pred_clipped - y_true))
    return likehoods

  # Backward pass
  def backward(self, dvalues, y_true):
    # Number of samples
    n_samples = dvalues.shape[0]
    # Adjust shape for calculation
    dvalues = dvalues.reshape(y_true.shape)
    # Calculate gradient
    self.dinputs = (dvalues - y_true) * 2
    # Normalize gradient
    self.dinputs = self.dinputs / n_samples