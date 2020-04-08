import tensorflow as tf
import keras
from keras.datasets import cifar10
import pandas as pd
import numpy as np


def get_data():
    train_bias = np.ones((50000,1))
    test_bias = np.ones((10000,1))
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.reshape(50000,3072)
    x_test = x_test.reshape(10000,3072)
    #normalize it here also!
    return (np.concatenate((train_bias,x_train), axis = 1), y_train), (np.concatenate((test_bias,x_test), axis = 1), y_test)

class ANN():
    def __init__(self,X,Y,units_count_hidden_layer,batch, activation = "sigmoid"):
        self.batch = batch
        if batch>0:
           self.X = np.split(X, batch)
           self.Y = np.split(Y, batch)
        else:
            self.batch = 1
            self.X = X
            self.Y = Y

        self.L = len(units_count_hidden_layer)+2 #input and output are the +2
        self.units_count_hidden_layer = units_count_hidden_layer

        #np.ones((len(X),1))
        self.bias_term = [1]*(self.L-1)
       
        self.w = [np.random.rand(len(X[0]),units_count_hidden_layer[0])] # old + 1 * new (the +1 is for the bias term)
        for i in range(self.L-1):
            self.w.append(np.random.rand(units_count_hidden_layer[i]+1,units_count_hidden_layer[i+1]) )
        self.w.append(np.random.rand(units_count_hidden_layer[-1]+1,len(Y[0])))
        # size of w is L -1
        #len(numpy) gives number of rows
        self.activation = activation
        self.act_units = [] #will be the same size as w eventually (the input units "have already been activated")

    def fit(self,alpha):
       self.mini_batch(alpha)

    def mini_batch(self,alpha):
        for i in range(self.batch):
            self.fwd_prop(i)
            self.back_prop(i,alpha)
            #self.gradient[1:] = (1/len(self.X[i]))*self.gradient[1:] #+ reguralization

    def fwd_prop(self,batch):
        for l in range(1,self.L):
            self.act_units.append(self.activate_layer(l,batch))

    def back_prop(self,batch,alpha):
        #total of L-1 updates to gradient
        dz = self.act_units[-1]-self.Y[batch]
        dw = np.dot( dz.T , self.act_units[-1] )/len(self.X)
        for l in range(2,self.L):
            dz = np.dot( dz , self.w[-(l+1)].T) *self.act_deriv(self.act_units[-l])
            dw = np.dot( self.act_units[-l].T, dz )/len(self.X)
            self.w[-(l+1)] -= alpha * dw # + reg

    def get_z(self,layer,batch): # layer is the layer we are creating 1<layer<L (not zero)
        if layer == 1:
            np.dot(self.X[batch],self.w[layer-1])
            self.act_units = [self.X[batch]]
        else:
            return np.dot(self.act_units[-1],self.w[layer-1])

    def activate_layer(self,layer,batch):
        if self.activation == "sigmoid":
            act = 1/(1+np.exp(-self.get_z(layer,batch)) )
        
        if layer<self.L-1: # incorporate the bias term now for the next layer (adding a bias col/cell to every row) adding a bias neuron
            #bias = np.full( (len(act),1),self.bias_term[layer]) # this value is edited in backprop so must
            bias = np.ones((len(act),1))
            return np.concatenate( (bias,act), axis = 1)
        else:
            return act
    def act_deriv(self,func):
        return func * (1.0 - func)

if __name__ == "__main__":
    #get_data()
    x = ANN(np.array([[2,1],[1]]),[2],[2,4,4,4],10)
    print()
    print(len(x.w))
    for i in x.w:
        #print(i.shape)
        print(i)