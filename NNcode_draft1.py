# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:05:29 2024

@author: xuyun
"""

import numpy as np
from scipy import optimize
import LinearModel as LF
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from datetime import datetime
from pytz import timezone

#%%
def ValueFunctionTerminal(gamma, C):
    return 1/gamma * np.power(C, gamma)

def StateActionValueFunctionTerminal(c1,y1,gamma,v, u ,r1,r2,r3,r4):
    temp_c = (u[0]*r1+u[1]*r2+u[2]*r3+u[3]*r4)*(c1+y1-u[4])
    value = 1/gamma * np.power(u[4], gamma) + v*ValueFunctionTerminal(gamma, temp_c)
    return value

#%%
def Predictor(c1,y1,gamma,v ,r1,r2,r3,r4, nnweights, inputscaler, outputscaler, scaleOutput = 1):
    
    # if scaleOutput = 0 then use sigmoid activation (to find optimal proportion) and no output scaler
    # x, d are (number, ) arrays 
    
    inputdata = np.concatenate((
                                c1.reshape(-1,1),
                                y1.reshape(-1,1)),
                                gamma.reshape(-1,1), 
                                v.reshape(-1,1),
                                r1.reshape(-1,1),
                                r2.reshape(-1,1),
                                r3.reshape(-1,1),
                                r4.reshape(-1,1),axis = 1)
    
    inputdata = inputscaler.transform(inputdata)
    layer1out = np.dot(inputdata, nnweights[0]) + nnweights[1]
    layer1out = tf.keras.activations.elu(layer1out).numpy()
    layer2out = np.dot(layer1out, nnweights[2]) + nnweights[3]
    layer2out = tf.keras.activations.elu(layer2out).numpy()
    layer3out = np.dot(layer2out, nnweights[4]) + nnweights[5]
    layer3out = tf.keras.activations.elu(layer3out).numpy()
    layer4out = np.dot(layer3out, nnweights[6]) + nnweights[7]
    
    if scaleOutput == 0:   # for policy apply sigmoid
        output = tf.keras.activations.sigmoid(layer4out).numpy() 
    if scaleOutput == 1:   # for value function apply output scaler
        output = outputscaler.inverse_transform(layer4out)
    
    return output

#%%

#actually what is a quantizer?

def StateActionValueFunction(c1,y1,gamma,v, u,r1,r2,r3,r4, nnweights, inputscaler, outputscaler, quantizer):
    
    # quantizer already scaled by sigma
    
    numWeights = len(quantizer[0])
    
    val = 1/gamma * np.power(u[4], gamma) \
            + v* Predictor(np.ones(numWeights) * (u[0]*r1+u[1]*r2+u[2]*r3+u[3]*r4)*(c1+y1-u[4]),
                           np.ones(numWeights) * y1,
                           np.ones(numWeights) * gamma,
                           np.ones(numWeights) * r1,
                           np.ones(numWeights) * r2,
                           np.ones(numWeights) * r3,
                           np.ones(numWeights) * r4,
                           nnweights, inputscaler, outputscaler)
    
    return np.sum(val.flatten() * quantizer[0])

#%%

def BuildAndTrainModel(c1_train, y1_train, gamma_train, v_train, r1_train, r2_train, r3_train, r4_train,
                       quantizer,
                       nn_dim = 6, node_num = 20, batch_num = 64, epoch_num = 3000, N=10,
                       initializer = TruncatedNormal(mean = 0.0, stddev = 0.05, seed = 0)):
        
    # Create training input and rescale
    numTrain = len(c1_train)
    
    input_train = np.concatenate((c1_train.reshape(-1,1),
                                  y1_train.reshape(-1,1),
                                  gamma_train.reshape(-1,1),
                                  v_train.reshape(-1,1),
                                  r1_train.reshape(-1,1),
                                  r2_train.reshape(-1,1),
                                  r3_train.reshape(-1,1), 
                                  r4_train.reshape(-1,1)),axis = 1) # (M_train, 6) array
    
    input_scaler = MinMaxScaler(feature_range = (0,1))
    input_scaler.fit(input_train)
    input_train_scaled = input_scaler.transform(input_train)
    
    valuefun_train = np.zeros((N+1, numTrain))
    policy_train = np.zeros((N+1, numTrain))
    
    
    # Create objects to save all NN solvers and scalers     
    output_scaler_valuefun = np.empty(N+1, dtype = object)
    nnsolver_valuefun = np.empty(N+1, dtype = object)
    nnsolver_policy = np.empty(N+1, dtype = object)
    
    def constraint1(u):
        return u[0] + u[1] + u[2] + u[3] - 1
    def constraint2(u,c):
        return c*0.3 - u[4]
    def initialguess(c):
        x0 = np.array([0.25, 0.25, 0.25, 0.25, c*0.3])
        return x0
    
    cons = [{'type': 'eq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2}]
    
    start = time.perf_counter() 
    
    
    # Run through all time steps backwards 
    for j in range(N-1, 0, -1):
        
        start_i = time.perf_counter()
        print("Time step " + str(j))
        
        
        # Create training output for value function and policy
        for i in range(numTrain):
            
            if j < (N-1):
                f_i = lambda u: StateActionValueFunction(c1_train[i], y1_train[i], 
                                    gamma_train[i], v_train[i], u, r1_train[i], r2_train[i], r3_train[i], r4_train[i],
                                    nnsolver_valuefun[j+1].get_weights(),
                                    input_scaler, output_scaler_valuefun[j+1], 
                                    quantizer)
            
            else:
                f_i = lambda u: StateActionValueFunctionTerminal(c1_train[i], y1_train[i], 
                                    gamma_train[i], v_train[i], u, r1_train[i], r2_train[i], r3_train[i], r4_train[i],
                                      quantizer)
                        
            solf_i = minimize(f_i, initialguess(c1_train[i]) , method='SLSQP', constraints=cons)

        
            policy_train[j][i] = solf_i.x 
            valuefun_train[j][i] = solf_i.fun
        
        end_i = time.perf_counter()
        print("     optimizations done: " + str(round((end_i-start_i)/60,2)) + " min.")
        

        start_i = time.perf_counter()
                
            
        # Build and train NN model for value function    
        output_scaler_valuefun[j] = MinMaxScaler(feature_range = (0,1))
        output_scaler_valuefun[j].fit(valuefun_train[j].reshape(-1, 1))
        valuefun_train_scaled = output_scaler_valuefun[j].transform(valuefun_train[j].reshape(-1,1))     
        
        nnsolver_valuefun[j] = Sequential()    
        nnsolver_valuefun[j].add(Dense(node_num, input_shape = (nn_dim,), activation = 'elu',
                                    kernel_initializer = initializer, bias_initializer = initializer))            
        nnsolver_valuefun[j].add(Dense(node_num, activation = 'elu',
                                    kernel_initializer = initializer, bias_initializer = initializer))
        nnsolver_valuefun[j].add(Dense(node_num, activation = 'elu',
                                    kernel_initializer = initializer, bias_initializer = initializer))
        nnsolver_valuefun[j].add(Dense(1, activation = None,
                                    kernel_initializer = initializer, bias_initializer = initializer))
        
        nnsolver_valuefun[j].compile(optimizer = 'adam', loss = 'mean_squared_error')
        nnsolver_valuefun[j].fit(input_train_scaled, valuefun_train_scaled,
                              epochs = epoch_num, batch_size = batch_num, verbose = 0)
        
        end_i = time.perf_counter()
        print("     train value function done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
        
        start_i = time.perf_counter()
        
        
        # Build and train NN model for policy
        nnsolver_policy[j] = Sequential()        
        nnsolver_policy[j].add(Dense(node_num, input_shape = (nn_dim,), activation = 'elu',
                                  kernel_initializer = initializer, bias_initializer = initializer))
        nnsolver_policy[j].add(Dense(node_num, activation = 'elu',
                                  kernel_initializer = initializer, bias_initializer = initializer))
        nnsolver_policy[j].add(Dense(node_num, activation = 'elu',
                                  kernel_initializer = initializer, bias_initializer = initializer))
        nnsolver_policy[j].add(Dense(1, activation = 'sigmoid',
                                  kernel_initializer = initializer, bias_initializer = initializer))
             
        nnsolver_policy[j].compile(optimizer = 'adam', loss = 'mean_squared_error')          
        nnsolver_policy[j].fit(input_train_scaled, policy_train[j].reshape(-1, 1),
                            epochs = epoch_num, batch_size = batch_num, verbose = 0)
        
        end_i = time.perf_counter()
        print("     train policy done: " + str(round((end_i-start_i)/60,2)) + " min.")
        
    
    end = time.perf_counter()
    duration = (end-start)/60

    print("Duration: " + str(duration) + " min.")
    
    return nnsolver_policy, nnsolver_valuefun, input_scaler, output_scaler_valuefun











                                    