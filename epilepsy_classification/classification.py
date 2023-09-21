from pyedflib import highlevel
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl
import os 
from tensorflow.keras import regularizers
import logging
import mne
import pandas as pd 
mne.set_log_level('ERROR')  # 'ERROR' level will suppress info and warnings


# Initialize an empty list to store the datasets
def get_no_epilepsy_dataset(num_dataset):
    main_folder = './no_epilepsy_edf/'
    no_epilepsy_dataset = None
    stop_processing = False
    analyzing_ch = ['F7', 'T4', 'F3', 'C3', 'P3', 'C4', 'T5', 'F4', 'P4', 'FP1', 'FP2', 'O1', 'O2', 'F8','T3', 'T6', 'A1', 'A2', 'T1', 'T2', 'FZ', 'CZ', 'PZ']
    indices = []
    

    # Walk through the directory structure
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            # Check if the file has a specific extension (e.g., '.csv', '.xlsx')
            if file.endswith('.edf'):
                # Create the full file path
                 file_path = os.path.join(root, file)
                 data = mne.io.read_raw_edf(file_path)
                 
                 logging.disable(logging.NOTSET)
                 indices = find_indices(data.ch_names,analyzing_ch)
                 signals = data.get_data() 
                 
                 ch_names_all = data.ch_names
                 data= signals[indices, :] 
                 
                 if data.shape[0] == len(analyzing_ch) and data.shape[1] > 10000:
                 #if data.shape[0] == 20 and data.shape[1] > 10000:

                     if no_epilepsy_dataset is None:
                         no_epilepsy_dataset = data[:, 0:10000, np.newaxis]
                     elif no_epilepsy_dataset.shape[2]< num_dataset:
                         no_epilepsy_dataset = np.concatenate((no_epilepsy_dataset, data[:, 0:10000, np.newaxis]), axis = 2)
                     else: 
                        stop_processing = True  # Set the flag to stop processing
                        break
                                     
        if stop_processing:
            break
                    
        
    return no_epilepsy_dataset

def find_indices(ch_names_all, ch_names):
    indices = []
    for ch in ch_names:    
        string = ch
        index = [i for i, item in enumerate(ch_names_all) if item.find(string) != -1]
        if len(index):
            indices.append(index[0])
        
    return indices 
    
# Initialize an empty list to store the datasets
def get_epilepsy_dataset(num_dataset):
    main_folder = './epilepsy_edf/'
    epilepsy_dataset = None
    stop_processing = False
    
    analyzing_ch = ['F7', 'T4', 'F3', 'C3', 'P3', 'C4', 'T5', 'F4', 'P4', 'FP1', 'FP2', 'O1', 'O2', 'F8', 'T3', 'T6', 'A1', 'A2', 'T1', 'T2', 'FZ', 'CZ', 'PZ']
    indices = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            # Check if the file has a specific extension (e.g., '.csv', '.xlsx')
            if file.endswith('.edf'):
                # Create the full file path
                 file_path = os.path.join(root, file)
                 
                 data = mne.io.read_raw_edf(file_path)

                 signals = data.get_data() 

                 indices = find_indices(data.ch_names,analyzing_ch)
                 signals = data.get_data() 
                 data= signals[indices, :] 
                 
                 if data.shape[0] == len(analyzing_ch)and data.shape[1] > 10000:
                 #if data.shape[0] == 20 and data.shape[1] > 10000:
    
                     if epilepsy_dataset is None:
                         epilepsy_dataset = data[:, 0:10000, np.newaxis]
                     elif epilepsy_dataset.shape[2]< num_dataset:
                         epilepsy_dataset = np.concatenate((epilepsy_dataset, data[:, 0:10000, np.newaxis]), axis = 2)
                     else: 
                         stop_processing = True  # Set the flag to stop processing
                         break
                              
        if stop_processing:
            break
                     
                        
    return epilepsy_dataset


def create_y_data(list_length, integer):
    y = None
    if integer == 0:
        y = np.zeros(list_length)
    elif integer == 1:
        y = np.ones(list_length)
        
    return y 
     
#choose number of datasets to use for each populatoin
num_dataset = 200
no_epilepsy = get_no_epilepsy_dataset(num_dataset)

epilepsy = get_epilepsy_dataset(num_dataset)

#create y value of 0 for no-epilepsy and 1 for epilepsy patients 
y_no_ep = create_y_data(no_epilepsy.shape[2], 0)
y_ep = create_y_data(epilepsy.shape[2], 1)

#choose ratio between training and testing dataset 
split_percentage = 0.7
num_split = round(split_percentage * num_dataset)

num_slices_no_epilepsy = no_epilepsy.shape[2]
random_indices_no_epilepsy= np.random.permutation(num_slices_no_epilepsy)

num_slices_epilepsy = epilepsy.shape[2]
random_indices_epilepsy= np.random.permutation(num_slices_epilepsy)

no_ep_vec_train = random_indices_no_epilepsy[0:num_split]
no_ep_vec_test = random_indices_no_epilepsy[num_split:]

ep_vec_train = random_indices_epilepsy[0:num_split]
ep_vec_test = random_indices_epilepsy[num_split:]


x_train = np.concatenate((no_epilepsy[:,:,no_ep_vec_train], epilepsy[:,:,ep_vec_train]), axis = 2)
x_test = np.concatenate((no_epilepsy[:,:,no_ep_vec_test], epilepsy[:,:,ep_vec_test]), axis= 2)

y_train = np.concatenate((y_no_ep[0:num_split], y_ep[0:num_split]), axis = 0)
y_test = np.concatenate((y_no_ep[num_split:], y_ep[num_split:]), axis = 0)

#randomize 
num_slices = x_train.shape[2]
random_indices_train = np.random.permutation(num_slices)
x_train = x_train[:,:,random_indices_train]
y_train = y_train[random_indices_train]


num_slices = x_test.shape[2]
random_indices_test = np.random.permutation(num_slices)
x_test = x_test[:,:,random_indices_test]
y_test = y_test[random_indices_test]

def calc_fft(x):
    N = x.shape[1]
    T = 1.0 /250.0
    yf = fft(x, axis = 1)
    xf = fftfreq(N, T)[:N//2]
    sp_yf = np.abs(yf[:,0:N//2])**2

    return sp_yf, xf


[sp_yf, xf] = calc_fft(x_train)
[sp_yf_test, xf_test] = calc_fft(x_test)

#extract features by getting spectral power in five frequency bands - alpha, beta, delta, gamma, theta 
def calc_freq_bands(sp_yf, xf):
        
    delta_ind = [i for i in range(len(xf)) if xf[i] > 0 and xf[i] < 4]
    theta_ind = [i for i in range(len(xf)) if xf[i] >= 4 and xf[i] <= 8]
    alpha_ind = [i for i in range(len(xf)) if xf[i] > 8 and xf[i] <= 12]
    beta_ind = [i for i in range(len(xf)) if xf[i] > 12 and xf[i] <= 30]
    gamma_ind = [i for i in range(len(xf)) if xf[i] > 30 and xf[i] < 125]
    
    delta = np.sum(sp_yf[:,delta_ind,:], axis = 1, keepdims = 1)
    theta = np.sum(sp_yf[:,theta_ind,:], axis = 1, keepdims = 1)
    alpha = np.sum(sp_yf[:,alpha_ind,:], axis = 1, keepdims = 1)
    beta = np.sum(sp_yf[:,beta_ind,:], axis = 1, keepdims = 1)
    gamma = np.sum(sp_yf[:,gamma_ind,:], axis = 1, keepdims = 1)
    
    tot = delta + theta + alpha + beta + gamma
    
    return [delta, theta, alpha, beta, gamma] 

#implement logistic regression with one layer
def logistic_reg(input_shape):
    input_img = tf.keras.Input(shape = input_shape)
    outputs = tfl.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(0.00001))(input_img)
    
    model = tf.keras.Model(inputs = input_img, outputs = outputs)
    
    return model 

def conv_net(input_shape):
    input_img = tf.keras.Input(shape = input_shape)
    Z1 = tfl.Conv2D(8, kernel_size = (4,4), padding = "same", kernel_regularizer=regularizers.L2(0.1))(input_img)

    A1 = tfl.ReLU()(Z1)
    
    P1 = tfl.MaxPool2D(pool_size = (8,8), strides = (8,8), padding = "same")(A1)

    Z2 = tfl.Conv2D(16, kernel_size = (2,2), padding = "same", kernel_regularizer=regularizers.L2(0.1))(P1)
    
    A2 = tfl.ReLU()(Z2)

    P2 = tfl.MaxPool2D(pool_size = (4,4), strides = (4,4), padding = "same")(A2)
   
    F = tfl.Flatten()(P2)
    
    outputs = tfl.Dense(1, activation = "sigmoid")(F)
    

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    
    return model 

def ANN(input_shape):
    alpha = 0.00001
    #relu -> dropout -> relu -> dropout 
    input_img = tf.keras.Input(shape = input_shape)

    Z1 =  tfl.Dense(128, activation = None)(input_img)
    
    Z1 = tf.keras.layers.Normalization(axis = None)(Z1)

    A1 = tfl.ReLU()(Z1)
    
    Z2 =  tfl.Dense(64, activation = None, kernel_regularizer=regularizers.L2(alpha))(A1)

    Z2 = tf.keras.layers.Normalization(axis = None)(Z2)

    A2 = tf.keras.layers.AlphaDropout(0.8)(Z2)
    
    Z3 =  tfl.Dense(32, activation = None, kernel_regularizer=regularizers.L2(alpha))(A2)

    Z3 = tf.keras.layers.Normalization(axis = None)(Z3)

    A3 = tfl.ReLU()(Z3)
    
    Z4 =  tfl.Dense(16, activation = None, kernel_regularizer=regularizers.L2(alpha))(A3)

    A4 = tfl.AlphaDropout(0.8)(Z4)
    
    Z5 =  tfl.Dense(8, activation = None, kernel_regularizer=regularizers.L2(alpha))(A4)

    outputs = tfl.Dense(1, activation = "sigmoid")(Z5)
    
    model = tf.keras.Model(inputs=input_img, outputs=outputs)

    return model 


def normalization(data):

    #data z-score normalization across features
    data_mean = np.mean(data, axis = 1)
    data_mean = np.reshape(data_mean, (data_mean.shape[0], 1))
    
    data_std = np.std(data, axis = 1)
    data_std = np.reshape(data_std, (data_std.shape[0], 1))
    
    data = (data - data_mean) / data_std
    
    return data


#data for convolutional neural network 
[delta, theta, alpha, beta, gamma] = calc_freq_bands(sp_yf, xf)
tot = delta + theta + alpha + beta + gamma

beta_gamma = np.multiply(beta, gamma)
delta_gamma = np.multiply(delta, gamma)
beta_theta = np.multiply(beta, theta)
[delta, theta, alpha, beta, gamma] = [delta, theta, alpha, beta, gamma] / tot

max_percent = np.max(np.concatenate([delta, theta, alpha, beta, gamma], axis = 1), axis = 1)
max_percent = max_percent[:,np.newaxis, :]

data_log = np.concatenate((beta_gamma, delta_gamma, beta_theta), axis = 0)
data_log = data_log.reshape((data_log.shape[2],data_log.shape[0]))

#data_log = normalization(data_log) 

#feature extraction  - obtain the relative frequency power and the product of frequency band values 
[delta_t, theta_t, alpha_t, beta_t, gamma_t] = calc_freq_bands(sp_yf_test, xf_test)
beta_gamma_t = np.multiply(beta_t, gamma_t) #product between beta and gamma 
delta_gamma_t = np.multiply(delta_t, gamma_t)#product between delta and gamma 
beta_theta_t = np.multiply(beta_t, theta_t)#product between beta and theta 
tot = delta_t + theta_t + alpha_t + beta_t + gamma_t

[delta_t, theta_t, alpha_t, beta_t, gamma_t] = [delta_t, theta_t, alpha_t, beta_t, gamma_t]/tot

#, beta_gamma_t, delta_gamma_t, beta_theta_t
#data_test_log = np.concatenate((delta_t, theta_t, alpha_t, beta_t, gamma_t, beta_gamma_t, delta_gamma_t, beta_theta_t), axis = 0)
data_test_log = np.concatenate((beta_gamma_t, delta_gamma_t, beta_theta_t), axis = 0)
data_test_log = data_test_log.reshape((data_test_log.shape[2],data_test_log.shape[0]))

#data_test_log = normalization(data_test_log) 

epoch_num = 700



input_shape = (data_log.shape[1]) #size must match with input (does not include the number of training examples)
ANN_model = ANN(input_shape)
optimizer = tf.keras.optimizers.RMSprop(learning_rate= 0.001)  # You can adjust the learning rate as needed
ANN_model.compile(optimizer = optimizer, loss='binary_crossentropy',metrics=['accuracy'])
ANN_model.summary()

train_dataset = tf.data.Dataset.from_tensor_slices((data_log, y_train)).batch(140)
test_dataset = tf.data.Dataset.from_tensor_slices((data_test_log, y_test)).batch(70)
history = ANN_model.fit(train_dataset, epochs=epoch_num, validation_data=test_dataset)

#plot the accuracy curve using the history information 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')