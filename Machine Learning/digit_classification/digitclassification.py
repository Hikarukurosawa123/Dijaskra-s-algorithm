#!/
#pip install sklearn


# In[12]:


import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import h5py
import sys
#np.set_printoptions(threshold=sys.maxsize)
import pandas as pd 
from numpy import genfromtxt
from testCases import *


# In[13]:


X = pd.read_csv("MNIST_data_train.csv")
X_data = X.values #convert to only values 

y = pd.read_csv("MNIST_target_train.csv")
y_data = y.values #convert to only values 



print(y_data)
#obtain maximum values for normalization 
X_max = np.max(X_data)
y_max = np.max(y_data)

#reshape data into (number of features, number of training examples) and normalize it 
X_data = X_data / X_max
y_data = y_data

#split to train and test data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data,y_data)

X_train = np.transpose(X_train)
X_test = np.transpose(X_test)


# In[ ]:





# In[14]:


def initialize_random(parameters, layers_dims, L):
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) 
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters 
    
def initialize_Xavier(parameters, layers_dims, L):
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.normal(0, 1/layers_dims[l-1], size = (layers_dims[l], layers_dims[l-1] )) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters 
    
def initialize_He(parameters, layers_dims, L):
    
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2./layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters 
    


# In[15]:


def initialize_parameters_deep(layers_dims):#initialize W and b 
    #np.random.seed(1)
    
    parameters = {}
    L = len(layers_dims)

    parameters = initialize_He(parameters, layers_dims, L)
    

    return parameters


# In[16]:


def initialize_velocity(parameters):
    L = len(parameters) //2
    v = {}
    s = {}
    
    for i in range(L):
        v["dW"+str(i + 1)] = np.zeros((parameters["W" + str(i+1)].shape))
        s["dW"+str(i + 1)] = np.zeros((parameters["W" + str(i+1)].shape))

        v["db"+str(i + 1)] = np.zeros((parameters["b" + str(i+1)].shape))
        s["db"+str(i + 1)] = np.zeros((parameters["b" + str(i+1)].shape))

    return v,s
        
    
def update_parameters_with_momentum(parameters, grads, v, s, t,epsilon = 1e-8, beta1 = 0.9, beta2 = 0.999, learning_rate = 0.01):
    L = len(parameters) //2
    v_corrected = {}
    s_corrected = {}
    
    for i in range(L):
        v["dW"+str(i + 1)] = beta1 * v["dW"+str(i + 1)] + (1-beta1) * grads["dW" + str(i+1)]
        v_corrected["dW"+str(i + 1)] = v["dW"+str(i + 1)] / (1-beta1**t)
        
        v["db"+str(i + 1)] = beta1 * v["db"+str(i + 1)] + (1-beta1) * grads["db" + str(i+1)]
        v_corrected["db"+str(i + 1)] = v["db"+str(i + 1)] / (1-beta1**t)

        s["dW"+str(i + 1)] = beta2 * s["dW"+str(i + 1)] + np.power(grads["dW" + str(i+1)], 2) * (1-beta2)
        s_corrected["dW"+str(i + 1)] = s["dW"+str(i + 1)] / (1-beta2**t)
        
        s["db"+str(i + 1)] = beta2 * s["db"+str(i + 1)] + np.power(grads["db" + str(i+1)], 2) * (1-beta2) 
        s_corrected["db"+str(i + 1)] = s["db"+str(i + 1)] / (1-beta2**t)
        
        
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * v_corrected["dW"+str(i + 1)] / (np.sqrt(s_corrected["dW"+str(i + 1)]) + epsilon)
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * v_corrected["db"+str(i + 1)] / (np.sqrt(s_corrected["db"+str(i + 1)]) + epsilon)

        
    return parameters, v, s


# In[17]:


def sigmoid(Z):
    cache = Z
    A = 1 / (1+np.exp(-Z))
    return A, cache

def RELU(Z):
    cache = Z
    return np.maximum(0,Z), cache

def softmax(Z): 
    cache = Z
    #print("softmax", np.exp(Z) / np.sum(np.exp(Z), axis = 0))
    return np.exp(Z) / np.sum(np.exp(Z), axis = 0), cache

    


# In[18]:


def linear_forward(A,W,b):
    Z = np.dot(W, A) + b
    cache = (A,W,b) #A is A_prev 

    return Z, cache 


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    #apply normalization 
    Z_mean = np.mean(Z, axis = 0, keepdims = 1)
    Z_var = np.var(Z, axis = 0, keepdims = 1)
    gamma = 1
    epsilon = 1e-8
    beta = 0

    Z = (Z - Z_mean) / np.sqrt(Z_var + epsilon)
    Z = gamma * Z + beta

    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)    
    elif activation == "relu":
        A, activation_cache= RELU(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
        #print("A_lin_Act", A)
        
    
    cache = (linear_cache, activation_cache)

    return A, cache


# In[19]:


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for i in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(i)], parameters["b" + str(i)], "relu")
        caches.append(cache)
    A_prev = A
    AL, cache = linear_activation_forward(A_prev, parameters["W" + str(L)], parameters["b" + str(L)], "softmax")
    caches.append(cache)

    return AL, caches

def one_hot_encoding(y):
    one_hot = np.zeros((int(y.max()) + 1, y.shape[0]))
    for x in range(y.shape[0]): #for each column 
        one_hot[int(y[x])][x] = 1
            
    #one_hot[y, np.arange(y.size)] = 1
    return one_hot


# In[20]:


def cross_entropy(y, y_hat):
    #print("y_hat, y", y_hat, y)
    return np.multiply(-np.log(y_hat),y) #- np.multiply((1-y), np.log(1-y_hat))
    
def compute_cost(A, y, parameters = None, lambd = 0.7):
    m = A.shape[1] #number of training examples
    #index = np.argmax(A, axis = 0)
    #print("this is the index", index)
    
    J = 1/m * np.sum(cross_entropy(y,A))
    return J



# In[ ]:





# In[21]:


def relu_grad(z):
    grad = z > 0
    return grad.astype(int)

def sigmoid_grad(z):
    return np.exp(-z) / np.power(1+np.exp(-z), 2)

def softmax_grad_forloop(z):
    # grad_Z_j = s_i * (1{i = j}  - s_j)
    num_classes = z.shape[0]
    A = softmax(z)
    Jacobian = np.zeros((num_classes, num_classes))
    indicator = 0
    for row in range(num_classes):
        for column in range(num_classes):
            if (row == column):
                indicator = 1
            else: 
                indicator = 0
            Jacobian[row][column] = A[row] * (indicator - A[column])
            
    return Jacobian 


# In[22]:


def relu_grad(z):
    grad = z > 0
    return grad.astype(int)

def sigmoid_grad(z):
    return np.exp(-z) / np.power(1+np.exp(-z), 2)

def softmax_grad(z):
    # grad_Z_j = s_i * (1{i = j}  - s_j)
    z_shape = z.shape
    num_classes = z_shape[0]
    A, cache = softmax(z)
    diagonal_matrices = A.reshape(z_shape[0],-1,1) * np.diag(np.ones(z_shape[1]))
    Outer = np.matmul(A.reshape(z_shape[0],-1,1), A.reshape(z_shape[0],1,-1))

    Jacobian = diagonal_matrices - Outer
            
    return Jacobian 


# In[23]:


def linear_backward(dZ, cache, lambd = 0.7):
    #takeas in linear cache
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dA_prev =np.dot(W.transpose(), dZ) 
    
    regularization = lambd / m * W #regularization term 
    dW = 1/m* np.dot(dZ, A_prev.transpose()) #+ regularization
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    return dA_prev, dW, db


# In[24]:


def linear_activation_backward(dA, cache, activation, Y = None, AL = None):
    linear_cache, activation_cache = cache # (A_prev,W,b) = linear_cache, Z = activation_cache
    if activation == "relu":
        dZ = np.multiply(dA, relu_grad(activation_cache))
    elif activation == "sigmoid":
        dZ = np.multiply(dA, sigmoid_grad(activation_cache))
    elif activation == "softmax":
        #dZ = np.multiply(dA, softmax(activation_cache))
        #print("AL here", AL)
        dZ = AL - Y

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    
    return dA_prev, dW, db


# In[25]:


def L_model_backward(AL, Y, caches):
    dA = -(np.divide(Y, AL) - (1-Y)/(1-AL))

    grads = {}
    L = len(caches)
    grads["dA" + str(L)] = dA

    #layer L 
    #print("caches", caches[L-1])
    dA_prev, dW, db = linear_activation_backward(dA, caches[L-1], "softmax", Y, AL = AL)
    grads["dA" + str(L-1)] = dA_prev
    grads["dW"  + str(L)] = dW
    grads["db" + str(L)] = db
    
    #layer L-1 ~ 
    for i in range(L-1, 0, -1):
        dA = dA_prev 
        dA_prev, dW, db = linear_activation_backward(dA, caches[i-1], "relu")
        grads["dA" + str(i-1)] = dA_prev
        grads["dW"  + str(i)] = dW
        grads["db" + str(i)] = db
        
    return grads 


# In[26]:


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    #parameters = params
    for x in range(0,L): #descent 
        parameters["W" + str(x+1)] = parameters["W" + str(x+1)] - learning_rate * grads["dW" + str(x+1)]
        parameters["b" + str(x+1)] = parameters["b" + str(x+1)] - learning_rate * grads["db" + str(x+1)]
        
    return parameters




# In[27]:


def gradient_check(dA, theta, Y, epsilon=1e-7, print_msg=True):
    """
    Implement the gradient checking presented in Figure 1.
    
    Arguments:
    x -- a float input
    theta -- our parameter, Z
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient. Float output
    """

    _, theta_plus = softmax(theta + epsilon )
    _, theta_minus = softmax(theta - epsilon )
    
    J_plus = compute_cost(theta_plus, Y)
    J_minus = compute_cost(theta_minus, Y)
    print("J_plus, J_minus", J_plus, J_minus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    
    grad = dA
    
    print("gradapprox", gradapprox)
    print("grad", grad)



# In[ ]:





# In[28]:


def L_layer_model(X,Y, layers_dims, parameters, learning_rate = 0.68, num_iterations = 1000, print_cost = False):
    #np.random.seed(1)
        
    v,s = initialize_velocity(parameters)

    costs = []
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y, parameters)
        grads = L_model_backward(AL, Y, caches)
        #grad_check = gradient_check(grads["dA2"], AL, Y)
        #parameters = update_parameters(parameters, grads, learning_rate)
        parameters, v,s = update_parameters_with_momentum(parameters, grads, v,s,t = 2, learning_rate = learning_rate)
        
        if print_cost and i % 10 == 0: #or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 10 == 0 or i == num_iterations:
            costs.append(cost)
    return parameters, costs

def pred(X, parameters, Y):
    AL, caches = L_model_forward(X,parameters)
    #print("softmax_val", AL[:, 0])
    index = np.argmax(AL, axis = 0)
    #print(index)
    accuracy = np.empty((Y.shape))
    #print(len(AL))
    for i in range(len(Y)):
        if(int(Y[i]) == index[i]):
            accuracy[i] = 1
        else: 
            accuracy[i] = 0

    print(accuracy)
    print("this is the sum", np.sum(accuracy) / len(accuracy))
    


# In[ ]:





# In[ ]:





# In[29]:


def random_mini_batches(X,Y, mini_batch_size = 64):
    np.random.seed(0)
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:, permutation]
    
    
    if m % mini_batch_size == 0:
        for i in range(0, m, mini_batch_size):
            mini_batches.append([shuffled_X[:, i:i+mini_batch_size], shuffled_Y[:, i:i+mini_batch_size]])

    else: 
        for i in range(0, m - mini_batch_size, mini_batch_size):

            mini_batches.append([shuffled_X[:, i:i+mini_batch_size], shuffled_Y[:, i:i+mini_batch_size]])
        final_index = int(m-mini_batch_size*np.floor(m/mini_batch_size))
        mini_batches.append([shuffled_X[:, m - final_index:], shuffled_Y[:, m - final_index:]])

    return mini_batches


# In[30]:


def training(X, Y, mini_batch_bool, num_epochs,  num_iterations, layers_dims):
    
    m = X.shape[0] # number of training examples
    if mini_batch_bool == 1:
        cost_total = 0
        parameters = initialize_parameters_deep(layers_dims)

        for i in range(num_epochs):
            mini_batches = random_mini_batches(X,Y)
            alpha = 0.46

            for x in range(len(mini_batches)):   
                parameters, costs = L_layer_model(mini_batches[x][0], mini_batches[x][1], layers_dims, parameters, learning_rate = alpha, num_iterations = num_iterations, print_cost = False)
                
    else: #if mini batch is not used
        #hyperparameter tuning 
        alphas = []
        for i in range(10):
            r = -4*np.random.rand()
            alpha = 10**r
            alphas.append(10 ** r)  
        parameters = initialize_parameters_deep(layers_dims)
            #alpha = 0.46
        alpha = 0.001
        parameters, costs = L_layer_model(X, Y, layers_dims, parameters, learning_rate = alpha, num_iterations = num_iterations, print_cost = True)
            
    return parameters, costs


# In[ ]:


#y_train = y_train.reshape(y_train.shape[1], y_train.shape[0])
#y_test = y_test.reshape(y_test.shape[1], y_test.shape[0])
print(y_train.shape)

one_hot_encoding = one_hot_encoding(y_train) #encoded vertically 

print("y_train", y_train)
print(y_train.shape)
print(one_hot_encoding.shape)

num_classes = one_hot_encoding.shape[0]
#note: y_input is expected to be hot-encoded 
layers_dims = [X_train.shape[0], 200, 80, num_classes] 

print("layers_dims", layers_dims)
#input y value is one-hot encoded 
parameters, costs = training(X_train, one_hot_encoding, 0, 10, 50, layers_dims)
#parameters, costs = L_layer_model(X_train, one_hot_encoding, layers_dims, print_cost = True, learning_rate = 0.68)
#y_input is expected to be a (number_of_training_examples, 1) vector 
pred(X_train, parameters, y_train) #training accuracy -> ~ 0.9145
pred(X_test, parameters, y_test) #testing accuracy -> ~ 0.916

# In[ ]:
iteration = np.arange(0, 201, 41) # -> plots iteration vs cost curve 
plt.plot(iteration, costs)
plt.xlabel("iteration")
plt.ylabel("cost")
plt.show()



