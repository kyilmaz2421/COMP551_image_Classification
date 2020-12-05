# Introduction
A Multi-layer Deep learning library with a highly configurable Neural Network built by hand

# USAGE example

    # can pull the cifar10 dataset by default to test library
    (x_train, y_train), (x_test, y_test) = get_cifar10_data()

    parameters = {
        'alpha': [0.00001],
        'regularization': [('L2',0.001)],
        'activation': ["relu"],
        'layers':[[100,10]],
        'momentum': [0.9],
        'batch': [128],
        'epcohs':[5]

    }

    clf = classifier(parameters)
    clf.search(x_train, y_train, validation_size = 0.1)
    clf.evaluate_model(x_test, y_test)

# how it works

Composed of two classes 'Classifier' and 'ANN' (artificial Neural Network). 

The ANN is the neural network class that has all the functionality to define a network with certain specifications and perform back propogation on a given datset.
It accepts a numpy matrix of training values X, a numpy vector of labels Y, a integer representing mini batch size, activation fucntion to use (i.e 'ReLu'), and an array where the length is the number of layers and the value at each index is the number of nodes in that layer

The Classifier is a wrapper class that interacts with the ANN class to more effectivley classidy data and perform hyperparamter tuning. The Classifier class gives the access to grid search, model evaluation, learning curves to gauge performance.

# improvments

The key improvments required are as defined below: 
- Implement dropout to mitigate overfitting 
- Develop and incorporate batch normalization 
- Maximize parrallel programming by using threads. Currently this library doesnt utilize multi-threading and would get a signifigant performance boost by being added.
- Design more tools for easily importing and normalizing any dataset
