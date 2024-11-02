import numpy as np


def loss(probs, y):
    """
    Calculate the softmax loss
    --------------------------
    :param probs: softmax probabilities
    :param y: correct labels
    :return: loss
    """
    loss = None

    #### Your implementation starts ######

    # compute the loss
    N = y.shape[0]  # Number of training data
    eps = 1e-8
    loss = -np.sum(np.log(probs[np.arange(N), y] + eps))  # total loss

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= N  # average loss

    ##### End of your implementation #####
    return loss
