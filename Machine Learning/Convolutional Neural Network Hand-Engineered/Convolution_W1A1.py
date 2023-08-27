#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
#from public_tests import *
import scipy
#from cnn_utils import *
#from test_utils import summary, comparator

import pandas as pd


# In[2]:


def one_hot_encoding(y):
    one_hot = np.zeros((int(y.max()) + 1, y.shape[0]))
    for x in range(y.shape[0]): #for each column 
        one_hot[int(y[x])][x] = 1
    return one_hot


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    #train set features
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:])  #  train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    #test set features
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(
        test_dataset["test_set_y"][:])  #  test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig[0:20, :, :, :]/255.
X_test = X_test_orig[0:20, :, :, :]/255.
Y_train = one_hot_encoding(Y_train_orig[0])
Y_test = one_hot_encoding(Y_test_orig[0])

Y_train = Y_train[:, 0:20]
Y_test= Y_test[:, 0:20]

index = 9
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


#%%


def zero_pad(X,pad):
    X_pad = np.pad(X, ((0,0), (pad, pad), (pad,pad), (0,0)), mode = "constant", constant_values = (0,0))
    
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    #assuming a_slice_prev has the same size as W and b
    Z = np.sum(np.multiply(a_slice_prev, W)) + float(b)
    
    
    return Z

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_c_prev) = A_prev.shape
    f, f, n_c_prev, n_c = np.shape(W)


    pad = hparameters["pad"]
    stride = hparameters["stride"]
    A_prev_pad = zero_pad(A_prev, pad)
    
    n_H = int(np.floor((n_H_prev - f + 2 * pad)/stride) + 1)
    n_W = int(np.floor((n_W_prev - f + 2 * pad)/stride) + 1)
    Z = np.zeros((m, n_H, n_W, n_c))

    for i in range(m):
        for layer in range(n_c):
            for column in range(n_W):
                horizontal_start = stride*column
                horizontal_end = f + stride*column
                for row in range(n_H):
                    vert_start = stride*row
                    vert_end = stride*row + f
                    Z[i,row,column, layer] = conv_single_step(A_prev_pad[i, vert_start:vert_end, horizontal_start:horizontal_end, :], W[:, :, :, layer], b[:, :, :, layer])                
                
    cache = (A_prev, W, b, hparameters)

    return Z, cache
    


# In[3]:


def pool_forward(A_prev, hparameters, mode = "max"):
    (m, n_H_prev, n_W_prev, n_c_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int(np.floor((n_H_prev - f)/stride)+1)
    n_W = int(np.floor((n_W_prev - f)/stride)+1)
    
    n_c = n_c_prev 
    
    A = np.zeros((m, n_H, n_W, n_c))
    
    for i in range(m):#number of layers  
        for layer in range(n_c): #number of filters 
            for column in range(n_W):
                horizontal_start = stride*column
                horizontal_end = f + stride*column
                for row in range(n_H):
                    vert_start = stride*row
                    vert_end = stride*row + f
                    if mode == "max":
                        A[i,row,column, layer] = np.max(A_prev[i, vert_start:vert_end, horizontal_start:horizontal_end, layer])
                    elif mode == "average":
                        A[i,row,column, layer] = np.mean(A_prev[i, vert_start:vert_end, horizontal_start:horizontal_end, layer])
                        
                        
    cache = (A_prev, hparameters)

        

    return A, cache


# In[4]:


def conv_backward(dZ, cache):
    A_prev, W, b, hparameters = cache
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    
    pad = hparameters["pad"]
    stride = hparameters["stride"]
    f,f, n_C_prev, n_C = W.shape
    
    m, n_H, n_W, n_C = dZ.shape

    
    dA_prev = np.zeros((A_prev.shape)) #expected size: m, n_H_prev, n_W_prev, n_C_prev 
    dW = np.zeros((W.shape)) #expected size: m, n_H, n_W, n_C
    db = np.zeros((1,1,1,n_C))
    
        
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):
        da_prev_pad = dA_prev_pad[i,:,:,:]
        a_prev_pad = A_prev_pad[i,:,:,:]
        for row in range(n_H):
            for column in range(n_W):
                for c in range(n_C):
                    vert_start = stride*row
                    vert_end = stride*row + f
                    horizontal_start = stride*column
                    horizontal_end = stride*column + f
                        
                    a_slice = a_prev_pad[vert_start:vert_end, horizontal_start:horizontal_end, :]

                    da_prev_pad[vert_start:vert_end, horizontal_start:horizontal_end, :] += W[:,:,:,c]*dZ[i,row, column,c]
                    dW[:,:,:,c] += a_slice * dZ[i,row,column,c]
                    db[:,:,:,c] += dZ[i,row,column,c]
                    
                    
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad, pad:-pad, :]
                    
    return dA_prev, dW, db


# In[5]:


def create_mask_from_window(x):
    
    max_val = np.max(x) #find the max value 
    
    mask = (x == max_val)
    
    return mask 

def distribute_value(dz, shape):
    
    row, col = shape
    average = np.sum(dz) / (row*col)
    a = np.multiply(np.ones((shape)), average)
    return a 

def pool_backward(dA, cache, mode = "max"):
    
    A_prev, hparameters = cache
    
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape 
    
    dA_prev = np.zeros((A_prev.shape))
    
    for i in range(m):
        a_prev = A_prev[i, :, :, :]
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    vert_start = stride*h 
                    vert_end =  stride*h + f
                    horiz_start = stride*w
                    horiz_end = stride*w + f
                    
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                        
                    elif mode == "average":
                        
                        # Get the value da from dA (≈1 line)
                        da = dA[i, h, w, c] # get single value dA from the matrix, distribute the contribution into a matrix
                        
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f,f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
                        # YOUR CODE STARTS HERE
                    
                    
    return dA_prev 


# In[6]:


def initialize_parameters_conv(f, n_layer, n_C, n_l_out): #n_C is the number of filters, n_layer = number of channels 
    #initialize parameters using He initialization, n_l and n_l+1 = 
    n_l = n_layer*f*f + n_l_out
    W = np.random.randn(f,f,n_layer, n_C) * np.sqrt(2./n_l) 
    b = np.zeros((1,1,1,n_C))
    
    cache = (W,b)
    return cache

def initialize_parameters(X, L):
    #hparameters layer 1 
    parameters = {}
    f1= 4
    stride1 = 1
    pad1 = 2 # same 
    
    m, n_H1, n_W1, n_L1 = X.shape
    num_filters_first = 8
    
    #hparameters layer 2
    f2= 2
    stride2 = 1
    pad2 = 2 # same 
    n_L2 = num_filters_first #number of layers
    num_filters_second = 16
    
    #hparameters layer 3
    n_H = 6;
    n_W = 64;

    
    #layer 1
    n_l_out = f2*n_L2*num_filters_second
    cache = initialize_parameters_conv(f1, n_L1,num_filters_first, n_l_out)
    parameters["W1"] = cache[0]
    parameters["b1"] = cache[1]

    cache = initialize_parameters_conv(f2, n_L2, num_filters_second, n_W)

    parameters["W2"] = cache[0]
    parameters["b2"] = cache[1]
    
    #final layer fully connected network - last layer has 6 nodes (because output is one of the 6 digits)
    cache = initialize_random(n_H, n_W)
    parameters["W3"] = cache[0]
    parameters["b3"] = cache[1]
    
    return parameters
    

def initialize_random(n_H, n_W):
    W = np.random.randn(n_H, n_W) * np.sqrt(2./n_W)
    b = np.zeros((n_H, 1))
    
    cache = (W,b)
    return cache 
def RELU(Z):
    cache = Z
    return np.maximum(0,Z), cache

def softmax(Z): 
    cache = Z
    return np.exp(Z) / np.sum(np.exp(Z), axis = 0), cache

def relu_grad(z):
    grad = z > 0
    return grad.astype(int)

def cross_entropy(y, y_hat):
    return np.multiply(-np.log(y_hat),y)

def compute_cost(A, y):
    m = A.shape[1] #number of training examples
    J = 1/m * np.sum(cross_entropy(y,A))
    return J

def linear_forward(A,W,b):
    Z = np.dot(W, A) + b
    cache = (A,W,b) #A is A_prev 

    return Z, cache 

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)    
    elif activation == "relu":
        A, activation_cache= RELU(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache


def linear_activation_backward(dA, cache, activation, Y, AL = None):
    linear_cache, activation_cache = cache # (A_prev,W,b) = linear_cache, Z = activation_cache
    if activation == "relu":
        dZ = np.multiply(dA, relu_grad(activation_cache))
    elif activation == "sigmoid":
        dZ = np.multiply(dA, sigmoid_grad(activation_cache))
    elif activation == "softmax":
        #dZ = np.multiply(dA, softmax(activation_cache))

        dZ = AL - Y

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    
    return dA_prev, dW, db

def linear_backward(dZ, cache, lambd = 0.7):
    #takeas in linear cache
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dA_prev =np.dot(W.transpose(), dZ) 
    
    regularization = lambd / m * W #regularization term 
    dW = 1/m* np.dot(dZ, A_prev.transpose()) #+ regularization
    db = 1/m * np.sfum(dZ, axis = 1, keepdims = True)
    
    return dA_prev, dW, db

def L_model_forward(X, parameters):
    #assigning parameters
    #parameters = {}
    caches = [] #appends results from the convolution or fully connected layers 
    pool_caches = [] #appends results from the pool layer 
    dA_caches = [] #for obtaining shapes of Z1 and Z2 for the shapes of dA_prevs
    #conv layer
    Z1_parameters = {}
    Z1_parameters["f"] = 4
    Z1_parameters["pad"] = 2
    Z1_parameters["stride"] = 1

    Z1, cache = conv_forward(X, parameters["W1"], parameters["b1"], Z1_parameters)
    caches.append(cache) # append cache containing (A_prev, W, b, hparameters)
    dA_caches.append(Z1.shape)

    
    #normalizing each feature map using z-score 
    output = Z1
    Z1_shape = Z1.shape  # (100, 65, 65, 8)

    # Reshape the output to collapse the training examples dimension
    reshaped_output = np.reshape(output, (Z1_shape[0], -1))

    mean_feature_maps = np.mean(reshaped_output, axis=1, keepdims = 1)
    var_feature_maps = np.var(reshaped_output, axis=1, keepdims = 1)

    mean_feature_maps = np.reshape(mean_feature_maps, (Z1_shape[0], 1, 1, 1))
    var_feature_maps = np.reshape(var_feature_maps, (Z1_shape[0], 1, 1, 1))
                              
                              
    print("shape", mean_feature_maps.shape)
    #print("mean", reshaped_mean)
    #print("var", reshaped_var)

    # make alpha1 and beta into trainable parameters
    epsilon = 1e-8
    alpha1 = 1
    beta = 0
    Z1_normalized = (output - mean_feature_maps)/ np.sqrt(var_feature_maps + epsilon)
    Z1_normalized_tilda = Z1_normalized * alpha1 + beta 
    
    #relu layer
    A1, cache = RELU(Z1_normalized_tilda)
    
    #pooling layer
    p1_parameters = {}
    p1_parameters["f"] = 8
    p1_parameters["stride"] = 8
    p1_parameters["pad"] = 2
    
    P1, cache = pool_forward(A1, p1_parameters, mode = "max")    
    pool_caches.append(cache)

    Z2_parameters = {}
    Z2_parameters["f"] = 2
    Z2_parameters["pad"] = 2
    Z2_parameters["stride"] = 1
    Z2, cache = conv_forward(P1, parameters["W2"], parameters["b2"], Z2_parameters)
    dA_caches.append(Z2.shape)
    
    #normalizing each feature map using z-score 
    output = Z2
    Z2_shape = Z2.shape  # (100, 65, 65, 8)

    # Reshape the output to collapse the training examples dimension
    reshaped_output = np.reshape(output, (Z2_shape[0], -1))

    mean_feature_maps = np.mean(reshaped_output, axis=1, keepdims = 1)
    var_feature_maps = np.var(reshaped_output, axis=1, keepdims = 1)

    mean_feature_maps = np.reshape(mean_feature_maps, (Z2_shape[0], 1, 1, 1))
    var_feature_maps = np.reshape(var_feature_maps, (Z2_shape[0], 1, 1, 1))
                              
    # make alpha1 and beta into trainable parameters
    epsilon = 1e-8
    alpha1 = 1
    beta = 0
    Z2_normalized = (output - mean_feature_maps)/ np.sqrt(var_feature_maps + epsilon)
    Z2_normalized_tilda = Z2_normalized * alpha1 + beta 

    caches.append(cache)
    A2, cache = RELU(Z2_normalized_tilda)
    
    p2_parameters = {}
    p2_parameters["f"] = 4
    p2_parameters["stride"] = 4
    p2_parameters["pad"] = 2
    
    P2, cache = pool_forward(A2, p2_parameters, mode = "max")
    pool_caches.append(cache)

    F2 = P2.reshape(P2.shape[0], -1).T #row - features, column - training examples

    #final layer fully connected network - last layer has 6 nodes (because output is one of the 6 digits)

    #activation function - softmax 
    A3, cache = linear_activation_forward(F2, parameters["W3"], parameters["b3"], activation = "softmax")

    caches.append(cache)
        
    return A3, caches, pool_caches, parameters, dA_caches

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

def linear_backward(dZ, cache, lambd = 0.7):
    #takeas in linear cache
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dA_prev =np.dot(W.transpose(), dZ) 

    regularization = lambd / m * W #regularization term 
    dW = 1/m* np.dot(dZ, A_prev.transpose()) #+ regularization
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)

    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    #parameters = params
    for x in range(0,L): #descent 
        
        parameters["W" + str(x+1)] = parameters["W" + str(x+1)] - learning_rate * grads["dW" + str(x+1)]
        parameters["b" + str(x+1)] = parameters["b" + str(x+1)] - learning_rate * grads["db" + str(x+1)]
        
    return parameters


def L_model_backward(AL, Y, caches, pool_caches, dA_caches):
    #conv2D -> RELU -> max_pool (1st layer) -> conv2D (2nd layer) -> RELU -> max_pool -> flatten -> softmax (3rd layer) 
    grads = {}
    L = len(caches)
    #softmax layer
    dA3 = -(np.divide(Y, AL) - (1-Y)/(1-AL))#dA for last layer 
    grads["dA" + str(L)] = dA3

    dA_prev_soft, dW, db = linear_activation_backward(dA3, caches[L-1], "softmax", Y, AL = AL)
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    grads["dA" + str(L-1)] = dA_prev_soft #dA of the fully connected neural network layer 
    
    #2nd pool layer
    cache = pool_caches[L-2] #obtain the input of the pool layer from the second layer
    dA_prev_soft = dA_prev_soft.T #reshape dA_prev, tranposing 
    dA_prev_soft = dA_prev_soft.reshape(dA_prev_soft.shape[0], 2,2,16) #converting back to the pre-flattened version - change to generalized version - P2.shape 
    dA_prev_pool = pool_backward(dA_prev_soft, cache) 
    
    #2nd RELU layer- reshape to get relu gradient, make it (number of features, number of training examples)
    dA_prev_pool = dA_prev_pool.reshape(dA_prev_pool.shape[0], -1).T
    dA_prev_relu = relu_grad(dA_prev_pool) #aka dZ for second layer
    
    #2nd conv layer - input - (dZ, cache), output = (A_prev, W, b, hparameters)
    cache = caches[L-2] #obtain cache containing (A_prev, W, b, hparameters)
    dA_prev_relu = dA_prev_relu.T
    dA_prev_relu = dA_prev_relu.reshape(dA_caches[1]) #  change to generalized version - Z2.shape
    dA_prev_conv, dW, db = conv_backward(dA_prev_relu, cache)
    
    grads["dW" + str(L-1)] = dW
    grads["db" + str(L-1)] = db
    grads["dA" + str(L-2)] = dA_prev_conv #dA of the fully connected neural network layer 
    
    #1st pool layer 
    cache = pool_caches[L-3] #obtain the input of the pool layer from the second layer
    dA_prev_pool = pool_backward(dA_prev_conv, cache)
    

    #1st relu layer - reshape to get relu gradient, make it (number of features, number of training examples)
    dA_prev_pool = dA_prev_pool.reshape(dA_prev_pool.shape[0], -1).T
    dA_prev_relu = relu_grad(dA_prev_pool) #aka dZ for first layer
    
    #1st conv layer
    cache = caches[L-3]
    dA_prev_relu = dA_prev_relu.T
    dA_prev_relu = dA_prev_relu.reshape(dA_caches[0]) #  change to generalized version - Z2.shape
    dA_prev_conv, dW, db = conv_backward(dA_prev_relu, cache)
    grads["dW" + str(L-2)] = dW
    grads["db" + str(L-2)] = db
    grads["dA" + str(L-3)] = dA_prev_conv #dA of the fully connected neural network layer 
        
    return grads 

def pred(X, parameters, Y):
    AL, caches, pool_caches, parameters, dA_caches = L_model_forward(X,parameters)
    index = np.argmax(AL, axis = 0)
    answer = np.argmax(Y, axis = 0)
    accuracy_vec = np.empty(len(answer))
    for i in range(len(answer)):
        if(answer[i] == index[i]):
            accuracy_vec[i] = 1
        else: 
            accuracy_vec[i] = 0
    accuracy = np.sum(accuracy_vec) / len(accuracy_vec)
    
    print("accuracy:", accuracy)
    
    
def convolutional_model(X, Y, learning_rate = 0.68, num_iterations = 50, print_cost = True):
    costs = []
    parameters = initialize_parameters(X_train, 3)
    v,s = initialize_velocity(parameters)

    for i in range(num_iterations):
        AL, caches, pool_caches, parameters, dA_caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches, pool_caches, dA_caches)
        learning_rates = [0.001, 0.01, 0.1]
        parameters, v,s = update_parameters_with_momentum(parameters, grads, v,s,t = 2, learning_rate = learning_rate)
        
        if print_cost and i % 10 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 10 == 0 or i == num_iterations:
            costs.append(cost)
    #cache
    return parameters, costs
    



alphas = [0.001, 0.01, 0.1]
parameters, costs = convolutional_model(X_train, Y_train, learning_rate = 0.1)
pred(X_train, parameters, Y_train)#[0][0:10]) -> training accuracy ~ 0.45 
pred(X_test, parameters, Y_test)#[0][0:10]) -> testing accuracy ~ 0.15


#%%

# plot 
iteration = np.arange(0, 41, 10) # -> plots iteration vs cost curve 
plt.plot(iteration, costs)
plt.xlabel("iteration")
plt.ylabel("cost")
plt.show()





