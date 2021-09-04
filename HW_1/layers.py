#coding=utf-8

# Library with layers for Technotrack task #1

## Layes
class Linear:
    def __init__(self, input_size, output_size, no_b=False):
        '''
        Creates weights and biases for linear layer from N(0, 0.01).
        Dimention of inputs is *input_size*, of output: *output_size*.
        no_b=True - do not use interception in prediction and backward (y = w*X)
        '''
        #### YOUR CODE HERE
        pass

    # N - batch_size
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        pass

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        #### YOUR CODE HERE
        pass

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        #### YOUR CODE HERE
        pass


## Activations
class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        pass

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        #### YOUR CODE HERE
        pass

    def step(self, learning_rate):
        pass

class ELU:
    def __init__(self, alpha):
        #### YOUR CODE HERE
        pass

    def forward(self, X):
        #### YOUR CODE HERE
        pass

    def backward(self, dLdy):
        #### YOUR CODE HERE
        pass

    def step(self, learning_rate):
        pass


class ReLU:
    def __init__(self, a):
        #### YOUR CODE HERE
        pass

    def forward(self, X):
        #### YOUR CODE HERE
        pass
      
    def backward(self, dLdy):
        #### YOUR CODE HERE
        pass

    def step(self, learning_rate):
        pass


class Tanh:
    def forward(self, X):
        #### YOUR CODE HERE
        pass

    def backward(self, dLdy):
        #### YOUR CODE HERE
        pass

    def step(self, learning_rate):
        pass


## Final layers, loss functions
class SoftMax_NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        #### YOUR CODE HERE
        #### (Hint: No code is expected here, just joking)
        pass

    def forward(self, X):
        '''
        Returns SoftMax for all X (matrix with size X.shape, containing in lines probabilities of each class)
        '''
        #### YOUR CODE HERE
        pass

    # y - true labels. Calculates dL/dy, returns dL/dX
    def backward(self, y):
        #### YOUR CODE HERE
        pass

class MSE_Error:
    # Saves X for backprop, X.shape = N x 1
    def forward(self, X):
        #### YOUR CODE HERE
        pass

    # Returns dL/dy (y - true labels)
    def backward(self, y):
        #### YOUR CODE HERE
        pass


## Main class
# loss_function can be None - if the last layer is SoftMax_NLLLoss: it can produce dL/dy by itself
# Or, for example, loss_function can be MSE_Error()
class NeuralNetwork:
    def __init__(self, modules, loss_function=None):
        '''
        Constructs network with *modules* as its layers
        '''
        #### YOUR CODE HERE
        pass
    
    def forward(self, X):
        #### YOUR CODE HERE
        #### Apply layers to input
        pass

    # y - true labels.
    # Calls backward() for each layer. dL/dy from k+1 layer should be passed to layer k
    # First dL/dy may be calculated directly in last layer (if loss_function=None) or by loss_function(y)
    def backward(self, y):
        #### YOUR CODE HERE
        pass

    # calls step() for each layer
    def step(self, learning_rate):
        pass