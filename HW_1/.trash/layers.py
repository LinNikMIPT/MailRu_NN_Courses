import numpy as np

class Linear:
    def __init__(self, input_size, output_size, no_b=False):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        #### YOUR CODE HERE
        #### Create weights, initialize them with samples from N(0, 0.1).
        self.input_size = input_size
        self.output_size = output_size
        
        self.no_b = no_b
        
        self.W = np.random.randn(self.input_size, self.output_size)*0.01 #Случайные коэффициенты с дисперсией 0.1
        self.b = np.random.randn(self.output_size)*0.01

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        self.X = X
        return X.dot(self.W)+self.b

    def backward(self, dLdy): #Считаем градиент на нейроне
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        self.dLdW = self.X.T.dot(dLdy)
        self.dLdb = dLdy.sum(0)
        self.dLdx = dLdy.dot(self.W.T)
        return self.dLdx

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - learning_rate*dLdw
        '''
        self.W = self.W - learning_rate * self.dLdW
        if not self.no_b:
            self.b = self.b - learning_rate * self.dLdb

class Sigmoid:
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.s = 1.0/(1+np.exp(-X))
        return self.s
    
    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return self.s*(1-self.s)*dLdy
    
    def step(self, learning_rate):
        pass

class ELU:
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, X):
        self.X = X
        self.mask = (X <= 0)
        result = X.copy()
        result[self.mask] = self.alpha*(np.exp(result[self.mask])-1)
        return result
      
    def backward(self, dLdy):
        result = dLdy.copy()
        
        result[self.mask != False] = dLdy[self.mask != False]
        result[self.mask] = self.alpha*np.exp(result[self.mask])*dLdy[self.mask]
        
        dydw = np.zeros(dLdy.shape)
        dydw[self.mask] = np.exp(self.X[self.mask])*dLdy[self.mask]
        self.dLda = np.sum(dLdy*dydw)
        return result

    def step(self, learning_rate):
        pass

class ReLU:
    def __init__(self, a):
        self.a = a

    def forward(self, X):
        self.X = X
        self.mask = (X <= 0)
        result = X.copy()
        result[self.mask] = self.a*(result[self.mask])
        return result
      
    def backward(self, dLdy):
        result = dLdy.copy()
        
        result[self.mask != False] = dLdy[self.mask != False]
        result[self.mask] = self.a*dLdy[self.mask]
        
        dydw = np.zeros(dLdy.shape)
        dydw[self.mask] = self.X[self.mask]*dLdy[self.mask]
        self.dLda = np.sum(dLdy*dydw)
        return result

    def step(self, learning_rate):
        pass


class Tanh:
    def forward(self, X):
        self.X = np.copy(X)
        return np.tanh(X)
      
    def backward(self, dLdy):
        return dLdy * (1 / (np.cosh(self.X) ** 2))

    def step(self, learning_rate):
        pass


class SoftMax_NLLLoss:
    # Рассчитывает последний узел и подготавливает данные для расчета ошибки
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        Returns SoftMax for all X
        '''
        #SoftMax
        self.p = np.exp(X)
        self.p /= self.p.sum(1, keepdims=True)
        return self.p
    
    # Это особый (конечный) слой => dLdy = y
    # Расчитывает производную фукции ошибки по входным данным конечного узла
    def backward(self, y):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        self.y = np.zeros(self.p.shape) #Соответствует форме X
        self.y[np.arange(self.p.shape[0]), y] = 1

        return (self.p - self.y) / self.p.shape[0]
    
    def step(self, learning_rate):
        pass


# Из-за того, что эти SoftMax и NLLLoss неразделимы, приходится предсусматривать в нейронной сети 2 возможности:
# работы с функцией потерь отдельно и в составе последнего нейрона
class NeuralNetwork:
    def __init__(self, modules, loss_function=None):
        '''
        Constructs network with *modules* as its layers
        loss_fuction - if the last neuron cannot calculate dLdy from y(_true)
        '''
        self.modules = modules
        self.loss_function = loss_function
    
    def forward(self, X):
        y_pred = X
        for i in range(len(self.modules)):
            y_pred = self.modules[i].forward(y_pred)
            
        # Функция потерь должна запомнить предсказания, чтобы потом по ним расчитать градиент
        if not (self.loss_function is None):
            self.loss_function.forward(y_pred)
        return y_pred
    
    def backward(self, y):
        '''
        y is true labels. The last neuron (or self.loss_function if avaliable) should calculate dLdy from y(_true)
        '''
        if not (self.loss_function is None):
            dLdy = self.loss_function.backward(y)
        else:
            dLdy = y
            
        for i in range(len(self.modules)-1, -1, -1): #Последний элемент не включается в range
            dLdy = self.modules[i].backward(dLdy)
    
    def step(self, learning_rate):
        for i in range(len(self.modules)):
            self.modules[i].step(learning_rate)

# MSE для векторов
class MSE_Error:
    # Получает результат работы нейросети и запоминает его для расчета функции ошибок
    def forward(self, X):
        self.X = X
    
    # Возвращает производную функции ошибок по результату работы нейросети
    def backward(self, y):
        return -2*(y[np.newaxis].T-self.X)/self.X.shape[0]