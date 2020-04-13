from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import cifar10
import pandas as pd
import numpy as np
from time import time


def get_data():
    train_bias = np.zeros((50000,1))
    test_bias = np.zeros((10000,1))
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.reshape(50000,3072)
    x_test = x_test.reshape(10000,3072) 
    print("Data retreived, normalized, and formatted succesfuly!\n")
    return (np.concatenate( (train_bias,x_train), axis = 1)/255, y_train), (np.concatenate( (test_bias,x_test), axis = 1)/255, y_test)

class classifier():
    def __init__(self,params): #Convolution or mlp (as a string)
        self.params = params
        self.clf = None
        self.best_param = None 

    def search(self,X,Y,validation_size,tolerance):
        t0 = time()
        keys = list(self.params.keys())
        key_len = len(keys)
        fit_count = 1
        for key in keys:
            fit_count *= len(self.params[key])
        print("Starting Grid Search....")
        print("Fitting "+str(fit_count)+" models")
        clfs = {} #{score:model}
        self.recurrence(X,Y,validation_size,{},self.params,keys,0,key_len,clfs,fit_count,tolerance)
        print("\ndone in %0.3fmins" % ((time() - t0)/60)+"\n")
        
        best_score = max(clfs.keys())
        self.best_param = clfs[best_score] #this is an array of all clfs with the best score
         
        print("Best Scores: ")
        print("For paramters: "+str(self.best_param)+"\nScore: "+ str(best_score)+"\n")
        #Re-fitting process for future use 
        self.clf = ANN(X,Y,self.best_param["layers"],self.best_param["batch"],self.best_param["activation"])
        self.clf.fit(self.best_param["alpha"], self.best_param["epochs"],self.best_param["regularization"])


    def recurrence(self,X,Y,validation_size,param,grid,keys,depth,max_depth,clfs,task_count,tolerance):  
        if depth == max_depth:
            if tolerance!=-1: 
                score = self.iterative_search(X,Y,param,validation_size,tolerance)
                param["epochs"] = score[3]
            else: 
                score = self.train_model(X,Y,validation_size,param,False)
            
            if clfs.get(score[0]): clfs[score[0]] = clfs[score[0]] = param
            else: clfs[score[0]] = param
            print("Score " +str(score[0])+ " for Param: "+str(param)+ " - (task: "+str(len(clfs.keys()))+ "/"+str(task_count) +")")
        else:
            for val in grid[keys[depth]]:
                param[keys[depth]] = val
                self.recurrence(X,Y,validation_size,param,grid,keys,depth+1,max_depth,clfs,task_count,tolerance)

    def train_model(self,X,Y,validation_size,params,evaluate,model=None):
        x_train,y_train,x_validate,y_validate = self.split_data(X,Y,validation_size)
        
        if model == None:
            clf = ANN(x_train,y_train,params["layers"],params["batch"],params["activation"])
        else: clf = model
            
        clf.fit(params["alpha"], params["epochs"],params["regularization"])
        if evaluate: self.evaluate_model(x_validate,y_validate,clf)
        return clf.eval_acc(clf.predict(x_validate),y_validate),clf.eval_acc(clf.predict(x_train),y_train)

    def split_data(self,X,Y,validation_size):
        data = np.concatenate((X,Y),axis=1)
        np.random.shuffle(data)
        test_split = int(validation_size*len(data))
        train,validation = data[test_split:], data[:test_split]#row split
        return train[:,0:-10],train[:,-10:],validation[:,0:-10],validation[:,-10:]

    def iterative_search(self,X,Y,params,validation_size,tolerance):
        x_train,y_train,x_validate,y_validate = self.split_data(X,Y,validation_size)
        clf = ANN(x_train,y_train,params["layers"],params["batch"],params["activation"])
        max_score,epochs,original_tolerance = 0,0,tolerance
        cv_score,train_score =[0],[0]
        improvement = True
        while improvement or tolerance>0: #while improving
            print("EPOCH: "+str(epochs)+" --> cv score "+str(cv_score[-1])+", train score "+ str(train_score[-1]))
            
            if not improvement: 
              tolerance-=1
              if tolerance <3 and params["alpha"]<0.01:# protect against local minima
                clf.mini_batch(np.split(x_train, params["batch"]),np.split(y_train, params["batch"]), params["alpha"]*10, params["regularization"])
              else: 
                clf.mini_batch(np.split(x_train, params["batch"]),np.split(y_train, params["batch"]), params["alpha"], params["regularization"])
            else: 
              tolerance = original_tolerance
              clf.mini_batch(np.split(x_train, params["batch"]),np.split(y_train, params["batch"]), params["alpha"], params["regularization"])

            
            cv_score.append(clf.eval_acc(clf.predict(x_validate),y_validate))
            train_score.append(clf.eval_acc(clf.predict(x_train),y_train))
            
            if (cv_score[-1]-max_score)>0.00001: 
              max_score = cv_score[-1]
              improvement = True
            else: improvement = False
            #shuffle data
            temp_data = np.concatenate((x_train,y_train),axis=1)
            np.random.shuffle(temp_data)
            x_train,y_train = temp_data[:,0:-10],temp_data[:,-10:]
            epochs +=1
       # self.evaluate_model(x_validate,y_validate,clf)
        self.plot_learning_curve(cv_score,train_score,np.arange(len(cv_score)),"Epochs")
        return max_score,cv_score,train_score,epochs-original_tolerance
            
    
    def evaluate_model(self,X_test,Y_test,clf):
        print("Evaluation of model:\n")
        Y_test = (np.argmax(Y_test,axis=1)+1)
        pred = clf.predict(X_test)
            
        print('Accuracy Score : ' + str(accuracy_score(Y_test,pred)))
        print('Precision Score : ' + str(precision_score(Y_test,pred, average='micro')))
        print('Recall Score : ' + str(recall_score(Y_test,pred, average='micro')))
        print('F1 Score : ' + str(f1_score(Y_test,pred, average='micro'))+"\n")
        #confusion matrix
        self.plot_cm(pred,Y_test)

    def plot_cm(self,pred,Y_test):
        print("Confusion Matrix")
        classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        cm = confusion_matrix(pred,Y_test)

        df_cm = pd.DataFrame(cm, index = classes,
                  columns = classes)
        sn.heatmap(df_cm)
        plt.show()

    def plot_learning_curve(self,cv_scores,test_scores,train_sizes,xlabel = "Training examples"):
        plt.figure()
        plt.title("title")
        plt.xlabel(xlabel)
        plt.ylabel("Score")
        cv_scores,test_scores = np.array([cv_scores]).T,np.array([test_scores]).T
        train_scores_mean = np.mean(cv_scores, axis=1)
        train_scores_std = np.std(cv_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()

    def learning_curve(self,X,Y,validation_size,train_sizes):
        if train_sizes == []: train_sizes = [0.33,0.66,1.0]
        cv_scores,test_scores = [],[]
        L = len(X)
        for t in train_sizes:
            split = int(t*L)
            split_x,split_y = X[:split],Y[:split]
            (cv,test) = self.train_model(split_x,split_y,validation_size,self.best_param,False)
            cv_scores.append(cv)
            test_scores.append(test)
        cv_scores,test_scores = np.array([cv_scores]).T,np.array([test_scores]).T
        self.plot_learning_curve(cv_scores,test_scores,train_sizes)

        

class ANN():
    def __init__(self,X,Y,hidden_layers,batch,activation):
        if (len(X)/batch).is_integer() and batch>0: self.batch = batch
        else: self.batch = 500

        self.data = np.concatenate((X,Y),axis=1)
        self.L = len(hidden_layers)+2 #input and output are the +2
        self.hidden_layers = hidden_layers
        
        self.activation = activation
        if (type(self.activation) == tuple and self.activation[0] == "Lrelu") or activation == "relu":
            self.scale_weights = lambda l : np.sqrt(2/l)
        else:
            self.scale_weights = lambda l : np.sqrt(1/l)

        #Initializing our weights
        #Note: columns (feautures) represent neurons
        self.w = [np.random.randn(len(X[0]),hidden_layers[0]+1)*self.scale_weights(len(X[0]))] # old + 1 * new (the +1 is for the bias term)
        for i in range(len(hidden_layers)-1): # size of w is L -1
            self.w.append(np.random.randn(hidden_layers[i]+1,hidden_layers[i+1]+1) *self.scale_weights(hidden_layers[i]) )
        self.w.append(np.random.randn(hidden_layers[-1]+1,len(Y[0])) *self.scale_weights(hidden_layers[-1]) )
        

    def fit(self, alpha, epochs,regularization):
        for i in range(epochs):
            np.random.shuffle(self.data)
            self.mini_batch(np.split(self.data[:,0:-10], self.batch),np.split(self.data[:,-10:], self.batch),alpha,regularization)

    def mini_batch(self,X,Y,alpha,regularization):
        for i in range(self.batch):
            #print("Batch "+str(i))
            act_units = [X[i]] # will be the same size as w eventually (the input units "have already been activated")
            self.fwd_prop(X[i], act_units)
            self.back_prop(X[i],Y[i],act_units,alpha,regularization)
    
    def fwd_prop(self,batch,act_units):
        #print("----Feed-forward starting")
        for l in range(1,self.L):
            new_layer = self.activate_layer(l,batch,act_units)
            act_units.append(new_layer)

    def activate_layer(self,layer,batch,act_units):
        if (layer == self.L-1) or (type(self.activation) != tuple and self.activation =="softmax"):
            z = self.get_z(layer,batch,act_units)
            #-np.max(z, axis=1, keepdims=True)
            act = np.exp(z)
            act = np.divide(act,np.sum(act,axis=1,keepdims=True)) # a value for every test set
        elif type(self.activation) == tuple and self.activation[0] == "Lrelu":
            z = self.get_z(layer,batch,act_units)
            act = np.maximum(0,z) + (self.activation[1] *np.minimum(0,z))
        elif self.activation == "sigmoid": #Sigmoid
            z = self.get_z(layer,batch,act_units)
            act = np.divide(1,(1+np.exp(-z )))
        elif self.activation=="tanh":
            z = self.get_z(layer,batch,act_units)
            act = np.divide(2,1+np.exp(-2*z))-1
        else: # Relu
            act = np.maximum(0,self.get_z(layer,batch,act_units))
        return act

    def get_z(self,layer,x_batch,act_units): # layer is the layer we are creating 1<layer<L (not zero)
        if layer == 1:
            return np.dot(x_batch,self.w[layer-1])
        else:
            return np.dot(act_units[layer-1],self.w[layer-1])


    def back_prop(self,X,Y,act_units,alpha,regularization):
        # total of L-1 updates to gradient
        dz = act_units[-1]-Y #self.act_units[-1] is our "y_hat"
        dw = np.dot( act_units[-2].T, dz )/len(X)
        self.w[-1] -= alpha * dw # + reg

        for l in range(2,self.L):
            dz = np.dot( dz , self.w[-l+1].T) * self.deriv(self.activation,act_units[-l])
            dw = np.dot( act_units[-(l+1)].T, dz )/len(X)
            self.w[-l] -= alpha * self.reguralize_dw(dw,regularization,l)
    
    def reguralize_dw(self,dw,regurlization,layer): #tuple (reg_type,[params])
        dw[1:] += regurlization[1]*self.w[-layer][1:]
        return dw
    
    def deriv(self,activation,act):
        if type(activation) == tuple and activation[0] == "Lrelu":
            vfunc = np.vectorize(lambda x : 1 if (x >= 0) else activation[1])
            return vfunc(act)
        elif activation == "relu":
            vfunc = np.vectorize(lambda x : 1 if (x >= 0) else 0)
            return vfunc(act)
        elif activation == "tanh": 
            return 1.0-np.square(np.divide(2,1+np.exp(-2*act))-1)
        else:
            return act * (1.0 - act) 

    def predict(self,x_test):
        prediction = [x_test]
        for l in range(1,self.L):
            new_layer = self.activate_layer(l,x_test,prediction)
            prediction.append(new_layer)
        return (np.argmax(prediction[-1],axis=1)+1)

    def eval_acc(self,pred,y):
        return np.sum(pred == (np.argmax(y,axis=1)+1) ) /len(y)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_data()

    parameters = {
        'alpha': [0.00001],
        'regularization': [('L2',0.1)],
        'activation': ["relu"],
        'layers':[[1]],
        'batch': [1],
    }

    clf = classifier(parameters)
    clf.search(x_train, y_train,0.1,6)
    #clf.evaluate_model(x_test, y_test) # on the tr
    #clf.learning_curve(x_train, y_train,0.1,[0.2,0.4,0.6,0.8,1.0])