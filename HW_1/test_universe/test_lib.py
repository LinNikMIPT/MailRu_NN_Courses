#coding=utf-8
# Система проверки 1го д/з по курсу Технотрека

import numpy as np
import matplotlib.pyplot as plt
import sys

def is_passed(task_name, condition):
    if condition:
        print(f"!!>> Task {task_name} passed")
    else:
        print(f"!!>> Task {task_name} NOT passed")

# 1-е задание - численный градиент
def check_gradient(func, X, gradient, eps = 1e-5):
    '''
    Computes numerical gradient and compares it with analytcal.
    func: callable, function of which gradient we are interested. Example call: func(X) - numeric function!
    X: np.array of size (n x m)
    gradient: np.array of size (n x m)
    Returns: maximum absolute diviation between numerical gradient and analytical.
    '''
    numerical_gradient = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] += eps
            y1 = func(X)
            X[i, j] -= 2*eps
            y2 = func(X)
            X[i, j] += eps
            numerical_gradient[i, j] = (y1-y2)/2/eps
    
    return (numerical_gradient - gradient).max()

def test_grad_checker(grad_checker):
    np.random.seed(1)
    EPS = 1e-7
    
    func = lambda x: (x**2).sum() # Хотим скалярную функцию ошибок, а check_gradient проверяет по очереди всю матрицу
    x = np.random.rand(10, 20)
    gradient = 2*x
    is_passed("test_grad_checker", grad_checker(func, x, gradient) < EPS)

def check_grads(linear_layer_class, sigmoid_layer_class, nll_sm_class, tanh_class):
    np.random.seed(2)
    EPS = 1e-7
    
    # Linear
    lin = linear_layer_class(18, 4)
    X = np.ones((23, 18))
    y = lin.forward(X)

    func = lambda x: lin.forward(x).sum() # Аналог функции ошибок
    dLdy = np.ones((23, 4)) # Очевидно, что производная суммы по всем y = 1
    gradient = lin.backward(dLdy)
    is_passed("check_linear_grad", check_gradient(func, X, gradient) < EPS)
    
    # Sigmoid
    X = np.ones((5, 10))
    s = sigmoid_layer_class()

    func = lambda x: s.forward(X).sum()
    dLdy = np.ones((5, 10))
    s.forward(X)
    gradient = s.backward(dLdy)
    is_passed("check_sigmoid_grad", check_gradient(func, X, gradient) < EPS)
    
    # Tanh
    X = np.random.randn(25, 10)
    t = tanh_class()
    func = lambda x: t.forward(X).sum()
    
    dLdy = np.ones((25, 10))
    t.forward(X)
    gradient = t.backward(dLdy)
    is_passed("check_tanh_grad", check_gradient(func, X, gradient) < EPS)
    
    # NLL_Softmax
    X = np.random.rand(15, 4)
    y = np.random.randint(0, 4, 15)
    loss = nll_sm_class()

    def func(x):
        p = loss.forward(x)
        y_tmp = np.zeros(p.shape) #Соответствует форме X
        y_tmp[np.arange(p.shape[0]), y] = 1
        return -(np.log(p)*y_tmp).sum(1).mean(0)

    loss.forward(X)
    is_passed("check_nll_grad", check_gradient(func, X, loss.backward(y)) < EPS)
    

# 3-е задание: MNIST
import layers

def pass_augmentations(X, augmentators):
    if augmentators is None:
        result = X.numpy()
    else:
        result = np.copy(X.numpy())
        for aug in augmentators:
            result = aug(result)
    return result.reshape((result.shape[0], -1))

def train(network, seed, epochs, learning_rate, train, test, check_level, name, augmentators=None):
    np.random.seed(seed)

    train_accuracy_epochs = []
    test_accuracy_epochs = []
    
    try:
        for epoch in range(epochs):
            accuracies = []

            for X, y in train:
                X = pass_augmentations(X, augmentators)
                y = y.numpy()

                prediction = network.forward(X)

                network.backward(y)
                network.step(learning_rate)
                accuracies.append((np.argmax(prediction, 1) == y).mean())
            train_accuracy_epochs.append(np.mean(accuracies))
            
            accuracies = []
            for X, y in test:
                X = pass_augmentations(X, augmentators)
                y = y.numpy()
                
                prediction = network.forward(X)
                accuracies.append((np.argmax(prediction, 1) == y).mean())
            test_accuracy_epochs.append(np.mean(accuracies))
            
            sys.stdout.write('\nEpoch {0}... Accuracy: {1:.3f}/{2:.3f}\n'.format(
                        epoch, train_accuracy_epochs[-1], test_accuracy_epochs[-1]))
            
            if test_accuracy_epochs[-1] > check_level:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\nBest for test: {0}'.format(max(test_accuracy_epochs)))
    
    plt.figure(figsize=(12, 5))
    plt.plot(train_accuracy_epochs, label='Train accuracy')
    plt.plot(test_accuracy_epochs, label='Test accuracy')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.grid('on')
    plt.show()
    
    is_passed(name, max(test_accuracy_epochs) > check_level)
    
def train_task3(learning_rate, train_loader, test_loader):
    neural_network = layers.NeuralNetwork([layers.Linear(784, 100), layers.Sigmoid(), layers.Linear(100, 100), 
                                           layers.Sigmoid(), layers.Linear(100, 10), layers.SoftMax_NLLLoss()])
    train(neural_network, 123, 15, learning_rate, train_loader, test_loader, 0.972, "task3_sigmoid_nn")
    
def train_task4_elu(a0_elu, learning_rate_elu, epochs_elu, train_loader, test_loader):
    neural_network = layers.NeuralNetwork([layers.Linear(784, 100), layers.ELU(a0_elu), layers.Linear(100, 100), 
                                           layers.ELU(a0_elu), layers.Linear(100, 10), layers.SoftMax_NLLLoss()])
    train(neural_network, 123, epochs_elu, learning_rate_elu, train_loader, test_loader, 0.977, "task4_elu")
    
def train_task4_relu(a0_relu, learning_rate_relu, epochs_relu, train_loader, test_loader):
    neural_network = layers.NeuralNetwork([layers.Linear(784, 100), layers.ReLU(a0_relu), layers.Linear(100, 100), 
                                           layers.ReLU(a0_relu), layers.Linear(100, 10), layers.SoftMax_NLLLoss()])
    train(neural_network, 123, epochs_relu, learning_rate_relu, train_loader, test_loader, 0.977, "task4_relu")
    
def train_task5(create_best_nn, create_best_augmentations, learning_rate, epochs, train_loader, test_loader):
    train(create_best_nn(), 123, epochs, learning_rate, train_loader, test_loader, 0.98, "task5", create_best_augmentations())