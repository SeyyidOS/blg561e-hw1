import numpy as np

def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    loss = None

    #### Your implementation starts ######

    # compute the loss
    N = # Number of training data
    loss = # total loss

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.

    loss = # average loss

    ##### End of your implementation #####
    return loss
