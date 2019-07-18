from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_examples = X.shape[0]
    num_classes = W.shape[1]
    
    scores = X.dot(W)
    scores -= np.max(scores, axis=1)[:, np.newaxis]
    probs = np.exp(scores)
    norm_probs = probs/(np.sum(probs,axis=1)[:,np.newaxis])
    
    #print(norm_probs.shape)
    for i in range(num_examples):
        loss -= np.log(probs[i, y[i]]/np.sum(probs[i,:]))
        
        dL_dy = norm_probs[i,:]
        dL_dy[y[i]] -= 1
        for c in range(num_classes):
            dW[:,c] += dL_dy[c]*X[i,:].T
        
    
    loss /= num_examples
    loss += reg*(np.sum(np.multiply(W, W)))
    
    # dL/dy = y - y_pred
    # y = Wx
    # dy/dW = x
    dW /= num_examples
    dW += reg*2*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_examples = X.shape[0]
    num_classes = W.shape[1]
    
    scores = X.dot(W)
    scores -= np.max(scores, axis=1)[:, np.newaxis]
    probs = np.exp(scores)
    norm_probs = probs/(np.sum(probs,axis=1)[:,np.newaxis])
    
    #print(norm_probs.shape)
    loss -= np.sum(np.log(probs[range(num_examples),y]/np.sum(probs, axis=1)))
    loss /= num_examples
    loss += reg*(np.sum(np.multiply(W, W)))
    
    mod_norm_probs = norm_probs
    mod_norm_probs[range(num_examples),y] -= 1
    dW = X.T.dot(mod_norm_probs)
    dW /= num_examples
    dW += reg*2*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
