from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.randn(input_dim, hidden_dim)*weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes)*weight_scale
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        N = X.shape[0]
        X_new = np.reshape(X, (N,-1))
        X2, cache2 = affine_relu_forward(X_new, self.params['W1'], self.params['b1'])
        a2, fc2_cache = affine_forward(X2, self.params['W2'], self.params['b2'])
        
        scores = a2
        
        #print(a2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dloss = softmax_loss(a2, y)
        loss += 0.5*self.reg*(np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2'])) 
        
        dx2, dw2, db2 = affine_backward(dloss, fc2_cache)
        _, dw1, db1 = affine_relu_backward(dx2, cache2)
        
        grads['W1'] = dw1 + 0.5*self.reg*2*self.params['W1']
        grads['b1'] = db1
        grads['W2'] = dw2 + 0.5*self.reg*2*self.params['W2']
        grads['b2'] = db2
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def affine_norm_relu_forward(x, w, b, gamma, beta, n_params, norm_type='batch_norm'):
    """
    Convenience layer that perorms an affine transform followed by a batchnorm, followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    
    if(norm_type == 'batchnorm'):
        #print('batchnorm running')
        a_norm, norm_cache, bn_params = batchnorm_forward(a, gamma, beta, n_params)
    else:
        a_norm, norm_cache = layernorm_forward(a, gamma.T, beta.T, n_params)
    
    out, relu_cache = relu_forward(a_norm)
    cache = (fc_cache, relu_cache, norm_cache)
    
    return out, cache, n_params


def affine_norm_relu_backward(dout, cache, norm_type='batchnorm'):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache, norm_cache = cache
    da = relu_backward(dout, relu_cache)
    
    if(norm_type == 'batchnorm'):
        da_norm, dgamma, dbeta = batchnorm_backward(da, norm_cache) 
        
    else:
        da_norm, dgamma, dbeta = layernorm_backward(da, norm_cache)
        
    dx, dw, db = affine_backward(da_norm, fc_cache)
    return dx, dw, db, dgamma, dbeta


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        for index in range(1, self.num_layers+1):
            # Let W.shape = (W_rows, W_cols)
            if(index == 1):
                W_rows = input_dim
                W_cols = hidden_dims[index-1]
                
                if(normalization=='batchnorm'):
                    self.params['gamma%i' % index] = np.asarray(1.0)
                    self.params['beta%i' % index] = np.asarray(0.0)
                    
                elif(normalization=='layernorm'):
                    self.params['gamma%i' % index] = np.ones((1,W_cols))
                    self.params['beta%i' % index] = np.zeros((1,W_cols))
                
            elif(index == self.num_layers):
                W_rows = hidden_dims[index-2]
                W_cols = num_classes
                
            else:
                W_rows = hidden_dims[index-2]
                W_cols = hidden_dims[index-1]

                if(normalization=='batchnorm'):
                    self.params['gamma%i' % index] = np.asarray(1.0)
                    self.params['beta%i' % index] = np.asarray(0.0)
                    
                elif(normalization=='layernorm'):
                    self.params['gamma%i' % index] = np.ones((1,W_cols))
                    self.params['beta%i' % index] = np.zeros((1,W_cols))
            
            self.params['W' + str(index)] = np.random.randn(W_rows, W_cols)*weight_scale
            self.params['b' + str(index)] = np.zeros(W_cols)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        N = X.shape[0]
        curr_X = np.reshape(X, (N,-1))
        caches = {}
        # num_layers includes the softmax layer (hidden_layers + 1)
        for layer_num in range(1,self.num_layers):
            w = self.params['W' + str(layer_num)]
            b = self.params['b' + str(layer_num)]
            
            if(self.normalization == 'batchnorm'):
                gamma = self.params['gamma%i' % layer_num]
                beta = self.params['beta%i' % layer_num]
                
                curr_X,curr_cache,curr_bn_params = affine_norm_relu_forward(curr_X, w, b, 
                                                                            gamma, beta, self.bn_params[layer_num-1],
                                                                            norm_type='batchnorm')
                self.bn_params[layer_num-1] = curr_bn_params
                
            elif(self.normalization == 'layernorm'):
                gamma = self.params['gamma%i' % layer_num]
                beta = self.params['beta%i' % layer_num]
                
                curr_X,curr_cache,curr_bn_params = affine_norm_relu_forward(curr_X, w, b, 
                                                                            gamma, beta, self.bn_params[layer_num-1],
                                                                            norm_type='layernorm')
                
            else:
                curr_X, curr_cache = affine_relu_forward(curr_X, w, b)
                
            
            if(self.use_dropout == True):
                curr_X, dropout_cache = dropout_forward(curr_X, self.dropout_param)
                
                if(self.normalization != None):
                    # un-pack and re-pack cache to include dropout mask
                    fc_cache, relu_cache, norm_cache = curr_cache

                    curr_cache = (fc_cache, relu_cache, norm_cache, dropout_cache)
                else:
                    fc_cache, relu_cache = curr_cache
                    curr_cache = (fc_cache, relu_cache, dropout_cache)
                
            caches[layer_num] = curr_cache
        
        w = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        curr_X, curr_cache = affine_forward(curr_X, w, b)
        caches[self.num_layers] = curr_cache
        scores = curr_X

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Compute data loss, initialize gradient dL/ds
        loss, dloss = softmax_loss(scores, y)
        
        # LAST AFFINE LAYER - Backpropagate gradient, compute reg loss 
        curr_dout, curr_dW, curr_db  = affine_backward(dloss, caches[self.num_layers])
        grads['W' + str(self.num_layers)] = curr_dW + self.reg*self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = curr_db
        
        loss += np.sum(self.params['W' + str(self.num_layers)]*self.params['W' + str(self.num_layers)])
        
        # AFFINE - RELU LAYERS - Backpropagate gradient, compute reg loss
        for layer_num in reversed(range(1, self.num_layers)):
            
            # Regularization Loss
            loss += 0.5*self.reg*np.sum(self.params['W' + str(layer_num)]*self.params['W' + str(layer_num)])
            
            # Drop-out Check
            if(self.use_dropout == True):
                if(self.normalization != None):
                    fc_cache, relu_cache, norm_cache, dropout_cache = caches[layer_num]
                    caches[layer_num] = (fc_cache, relu_cache, norm_cache)
                
                else:
                    fc_cache, relu_cache, dropout_cache = caches[layer_num]
                    caches[layer_num] = (fc_cache, relu_cache)
                    
                curr_dout = dropout_backward(curr_dout, dropout_cache)
                
                
            if(self.normalization == 'layernorm'):
                dx, curr_dW, curr_db, dgamma, dbeta  = affine_norm_relu_backward(curr_dout, caches[layer_num],
                                                                                 norm_type='layernorm')
                grads['gamma%i' % layer_num] = dgamma
                grads['beta%i' % layer_num] = dbeta
                
            elif(self.normalization == 'batchnorm'):
                dx, curr_dW, curr_db, dgamma, dbeta  = affine_norm_relu_backward(curr_dout, caches[layer_num])
                grads['gamma%i' % layer_num] = dgamma
                grads['beta%i' % layer_num] = dbeta
                
            else:
                dx, curr_dW, curr_db  = affine_relu_backward(curr_dout, caches[layer_num])
                
            grads['W' + str(layer_num)] = curr_dW + self.reg*self.params['W' + str(layer_num)]
            grads['b' + str(layer_num)] = curr_db
            curr_dout = dx
        
        #dx2, dw2, db2 = affine_backward(dloss, fc2_cache)
        #_, dw1, db1 = affine_relu_backward(dx2, cache2)
        
        #grads['W1'] = dw1 + 0.5*self.reg*2*self.params['W1']
        #grads['b1'] = db1
        #grads['W2'] = dw2 + 0.5*self.reg*2*self.params['W2']
        #grads['b2'] = db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
