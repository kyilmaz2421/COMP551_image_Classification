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
    def __init__(self,cv,params): #Convolution or mlp (as a string)
        #self.model = model
        self.cv = cv
        self.params = params
        self.clf = None
        self.best_param = None 

    def search(self,X,Y):
        t0 = time()
        keys = list(self.params.keys())
        key_len = len(keys)
        fit_count = 1
        for key in keys:
            fit_count *= len(self.params[key])
        print("Starting Grid Search....")
        print("Fitting "+str(self.cv)+" folds for each of the "+str(fit_count) +" parameters --> totalling in "+str(fit_count*self.cv)+" fits")
        clfs = {} #{score:model}
        self.recurrence(X,Y,self.cv,{},self.params,keys,0,key_len,clfs,fit_count)
        print("\ndone in %0.3fmins" % ((time() - t0)/60)+"\n")
        
        best_score = max(clfs.keys())
        self.best_param = clfs[best_score] #this is an array of all clfs with the best score
         
        print("Best Scores: ")
        print("For paramters: "+str(self.best_param)+"\nScore: "+ str(best_score)+"\n")
        #Re-fitting process for future use 
        self.clf = ANN(X,Y,self.best_param["layers"],self.best_param["batch"],self.best_param["activation"])
        self.clf.fit(self.best_param["alpha"], self.best_param["epochs"],self.best_param["regularization"])


    def recurrence(self,X,Y,k,param,grid,keys,depth,max_depth,clfs,task_count):  
        if depth == max_depth:
            score = self.k_cross_validation(X,Y,k,param)
            if clfs.get(score): clfs[score[0]] = clfs[score[0]] = param
            else: clfs[score[0]] = param
            print("Score " +str(score[0])+ " for Param: "+str(param)+ " - (task: "+str(len(clfs.keys()))+ "/"+str(task_count) +")")
        else:
            for val in grid[keys[depth]]:
                param[keys[depth]] = val
                self.recurrence(X,Y,k,param,grid,keys,depth+1,max_depth,clfs,task_count)

    def k_cross_validation(self,X,Y,k,params):
        if not (len(X)/k).is_integer(): 
            k = 5
        jump = (len(X)/k)
        cv_scores,test_scores = [],[]
        x_set,y_set = np.array_split(X,k),np.array_split(Y,k)
        for i in range(len(x_set)): #i is test
            print("----Completing cv fit "+str(i+1))
            left,right = int(jump*(i+1)),int(jump*i)
            x,y = np.concatenate( (X[:right],X[left:]), axis=0), np.concatenate( (Y[:right],Y[left:]), axis=0)
            clf = ANN(x,y,params["layers"],params["batch"],params["activation"])
            clf.fit(params["alpha"], params["epochs"],params["regularization"])
            cv_scores.append(clf.eval_acc(clf.predict(x_set[i]),y_set[i]))
            test_scores.append(clf.eval_acc(clf.predict(x),y))
        return np.average(cv_scores),np.average(test_scores)
    
    def eval_on_test(self,X_test,Y_test):
        print("Evaluation on test set:\n")
        Y_test = (np.argmax(Y_test,axis=1)+1)
        pred = self.clf.predict(X_test)
            
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

    def learning_curve(self,X,Y,train_sizes):
        if train_sizes == []: train_sizes = [0.33,0.66,1.0]
        cv_scores,test_scores = [],[]
        plt.figure()
        plt.title("title")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        L = len(X)
        for t in train_sizes:
            split = int(t*L)
            split_x,split_y = X[:split],Y[:split]
            (cv,test) = self.k_cross_validation(split_x,split_y,self.cv,self.best_param)
            cv_scores.append(cv)
            test_scores.append(test)

        cv_scores,test_scores = np.array([cv_scores]).T,np.array([test_scores]).T
        train_scores_mean = np.mean(cv_scores, axis=1)
        train_scores_std = np.std(cv_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()

        

class ANN():
    def __init__(self,X,Y,hidden_layers,batch,activation):
        if (len(X)/batch).is_integer() and batch>0: self.batch = batch
        else: self.batch = 5

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
        self.w = [np.random.rand(len(X[0]),hidden_layers[0]+1)*self.scale_weights(len(X[0]))] # old + 1 * new (the +1 is for the bias term)
        for i in range(len(hidden_layers)-1): # size of w is L -1
            self.w.append(np.random.rand(hidden_layers[i]+1,hidden_layers[i+1]+1) *self.scale_weights(hidden_layers[i]) )
        self.w.append(np.random.rand(hidden_layers[-1]+1,len(Y[0])) *self.scale_weights(hidden_layers[-1]) )
        

    def fit(self, alpha, epochs,regularization):
        for i in range(epochs):
            np.random.shuffle(self.data)
            self.mini_batch(alpha,regularization)

    def mini_batch(self,alpha,regularization):
        X = np.split(self.data[:,0:-10], self.batch)
        Y = np.split(self.data[:,-10:], self.batch)
        for i in range(self.batch):
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
        'alpha': [0.1],
        'regularization': [('L2',1)],
        'activation': [ ("Lrelu",0.1)],
        'layers':[[1000,1000]],
        'batch': [100],
        'epochs': [10]
    }

    clf = classifier(5,parameters)
    clf.search(x_train, y_train)
    #clf.eval_on_test(x_test, y_test)
    clf.learning_curve(x_train, y_train,[])