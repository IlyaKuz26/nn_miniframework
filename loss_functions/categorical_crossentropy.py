import numpy as np

from loss import Loss


class CategoricalCrossentropy(Loss):
  # Forward pass
  def forward(self, y_pred, y_true):
    n_samples = y_pred.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    if len(y_true.shape) == 1:
      confidences = y_pred_clipped[range(n_samples), y_true]
    elif len(y_true.shape) == 2:
      confidences = np.sum(y_pred_clipped * y_true, axis=1)

    negative_log_likehoods = -np.log(confidences)
    return negative_log_likehoods

  # Backward pass
  def backward(self, dvalues, y_true):
    # Number of samples
    n_samples = dvalues.shape[0]
    # Number of labels in every sample
    n_labels = dvalues.shape[1]

    # If labels are discrete, turn them into one-hot vectors
    if len(y_true.shape) == 1:
      y_true = np.eye(n_labels)[y_true]

    # Calculate gradient
    self.dinputs = -y_true / dvalues
    # Normalize gradient
    self.dinputs = self.dinputs / n_samples