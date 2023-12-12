import numpy as np


class Loss:
  def calculate(self, y_pred, y_true):
    samples_losses = self.forward(y_pred, y_true)
    data_loss = np.mean(samples_losses)
    return data_loss