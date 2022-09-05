# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:44:26 2020

@author: E1289292
"""
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda, Flatten, TimeDistributed#, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import merge
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from keras.models import load_model
import time
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=-1, keepdims=True))
hunits=10
model = load_model('C:/Users/E1289292/Downloads//SiameseLSTM10.h5',custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance})
model.summary()
inp = Input(batch_shape= (1,700, 1))
rnn= LSTM(hunits, return_sequences=True, name="RNN",unroll=True)(inp)
states = Model(inputs=[inp],outputs=[rnn])
states.summary()
states.layers[1].set_weights(model.layers[2].get_weights())  
test=np.random.randint(0, 100, size=(1, 700, 1))
start=time.time()
pred=states.predict(test)
end=time.time()
print("time cost:",end-start)
weights=model.layers[2].get_weights()
def sigmoid(x):
    return(1.0/(1.0+np.exp(-x)))
def LSTMlayer(weight,x_t,h_tm1,c_tm1):
    '''
    c_tm1 = np.array([0,0]).reshape(1,2)
    h_tm1 = np.array([0,0]).reshape(1,2)
    x_t   = np.array([1]).reshape(1,1)
    
    warr.shape = (nfeature,hunits*4)
    uarr.shape = (hunits,hunits*4)
    barr.shape = (hunits*4,)
    '''
    warr,uarr, barr = weight
    s_t = (x_t.dot(warr) + h_tm1.dot(uarr) + barr)
    hunit = uarr.shape[0]
    i  = sigmoid(s_t[:,:hunit])
    f  = sigmoid(s_t[:,1*hunit:2*hunit])
    _c = np.tanh(s_t[:,2*hunit:3*hunit])
    o  = sigmoid(s_t[:,3*hunit:])
    c_t = i*_c + f*c_tm1
    h_t = o*np.tanh(c_t)
    return(h_t,c_t)
    
c_tm1 = np.array([0]*hunits).reshape(1,hunits)
h_tm1 = np.array([0]*hunits).reshape(1,hunits)
output=[]
start=time.time()
for i in range(len(test[0])):
    x_t = test[0][i].reshape(1,1)
    h_tm1,c_tm1 = LSTMlayer(weights,x_t,h_tm1,c_tm1)
    #output.append(h_tm1)
    if i==0:
        output=h_tm1
    else:
        output=np.append(output, h_tm1,axis = 0) 
#output=output.reshape(-1,700,20)    
end=time.time()
print("time cost:",end-start)

