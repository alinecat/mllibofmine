# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:06:53 2022


Neural network training infrastructure
@author: Ali Necat
"""
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np

# mu, sigma = 0, 0.1 # mean and standard deviation
# s = np.random.normal(3, 2.5, size=(2, 4))


def rnn_example_training():
    x = np.linspace(-np.pi*5, np.pi*5, 200)
    y = np.sin(x)


    x = np.expand_dims(x, axis=1)

    y = np.expand_dims(y, axis=1)

    model = Model('mean_square')

    model.add_layer(RNN(200,200,50,5))
    
    #3 epochs is an approximation, 5 is better.
    model.train(x, y, 3, 0.0001)

    y_head = model.predict(x)

    plt.plot(x, y)
    plt.plot(x, y_head)
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    plt.axis('tight')
    plt.show()

def example_training():

    np.random.seed(1)

    mean1 = [1.0, 1.0]
    mean2 = [-1.0, -1.0]
    mean3 = [-1.0, 1.0]
    mean4 = [1.0, -1.0]
    cov = np.identity(2)*0.08

    class1 = np.concatenate((np.random.multivariate_normal(
        mean1, cov, 100), np.random.multivariate_normal(mean2, cov, 100)), axis=0)

    class2 = np.concatenate((np.random.multivariate_normal(
        mean3, cov, 100), np.random.multivariate_normal(mean4, cov, 100)), axis=0)

    X = np.concatenate((class1, class2), axis=0)
    # Y = np.concatenate((np.zeros(len(class1)),np.ones(len(class2))), axis = 0)

    Y = []
    aa = np.array([1, 0])
    for i in range(len(class1)):
        Y.append(aa)
    aa = np.array([0, 1])
    for i in range(len(class2)):
        Y.append(aa)
    Y = np.array(Y)
    plt.plot(class2.T[0], class2.T[1], '.')
    plt.plot(class1.T[0], class1.T[1], '.')

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    X, Y = unison_shuffled_copies(X, Y)

    X = X.T

    X = X/(np.max(abs(X)))

    model = Model('cross_entropy')
    model.add_layer(DenseLayer(2, 5, 'sigmoid'))
    # model.add_layer(DenseLayer(20,20,'sigmoid'))
    model.add_layer(DenseLayer(5, 2, 'sigmoid'))
    # model.train(X,Y,10000,1)
    model.train(X, Y, 15000, 1)
    pred = model.predict(X)
    pred = (pred > 0.5)*1
    pred = pred.T
    pred == Y
    acc = np.sum((pred == Y)*1)/800

    print("Accuracy is:", acc)


def softmax(act):
    softmx = np.exp(act)/np.sum(np.exp(act))
    return softmx

def tanh(act):
    tanhyp = np.sinh(act)/np.cosh(act)
    return tanhyp

def sigmoid(act):
    sigm = 1/(1+np.exp(-act))
    return sigm

def relu(act):
    return np.maximum(0, act)


class Model:
    def __init__(self, cost_fnc):
        self.layer_cnt = 0
        self.cost_fnc = cost_fnc
        self.layers = []
        self.input_size = 0
        self.output_size = 0

    def add_layer(self, layer):
        if (self.layer_cnt == 0):  # No layer added
            self.input_size = layer.input_size
            self.output_size = layer.output_size
            self.layer_cnt = self.layer_cnt + 1
            self.layers.append(layer)

        else:
            if (self.output_size != layer.input_size):
                raise Exception("Model Size Is Not Compatible")
            else:
                self.output_size = layer.output_size
                self.layer_cnt = self.layer_cnt + 1
                self.layers.append(layer)

    def predict(self, Input):
        for layer in self.layers:
            Input = layer.forward(Input)
        return Input

    def train(self, X, Y, epochs, learning_rate):  # Fit X to Y
        # Make a forward pass on the dataset
        # Get the loss for each input in dataset
        # Make backward pass for each input in dataset
        # Go unitl convergence

        for epoch in range(epochs):
            # for idx ,inp in enumerate(X):
            # forward_propagation
            inp = X
            for layer in self.layers:
                inp = layer.forward(inp)

            if self.cost_fnc == 'cross_entropy':
                # Calculate loss, categorical crossentropy
                cost = (-1/len(Y)) * \
                    (np.sum(Y.T*np.log(inp)+(1-Y.T)*np.log(1-inp)))
                print("Cost is:", cost)
                # Calculate the 1st gradient

                grad = - (np.divide(Y.T, inp) - np.divide(1 - Y.T, 1 - inp))
                # derivative of cost with respect to AL
                # print(grad)
            elif self.cost_fnc == 'mean_square':
                #print(np.shape(inp))
                cost = np.sum(np.square(Y - inp))/(2*len(Y))
                print("Cost is:", cost)
                grad = -(Y - inp)
                

            for layer in reversed(self.layers):
                if layer.type == 'dense':
                    dw, db, grad = layer.backward(grad)
                    layer.weights = layer.weights - learning_rate*dw
                    layer.bias = layer.bias - learning_rate*db
                elif layer.type == 'rnn':
                    d_inp_weights, d_recurrent_weights, d_out_weights = layer.backward(grad, X)
                    layer.inp_weights = layer.inp_weights - learning_rate*d_inp_weights
                    layer.recurrent_weights = layer.recurrent_weights - learning_rate*d_recurrent_weights
                    layer.out_weights = layer.out_weights - learning_rate*d_out_weights








class LSTM:
    def __init__(self, timesteps, output_size, hidden_units, bptt_unfold):
        
        self.forget_weights = np.random.randn(hidden_units, timesteps)*0.01
        self.input_weights = np.random.randn(hidden_units, hidden_units)*0.01
        self.output_weights = np.random.randn(output_size, hidden_units)*0.01
        self.out = np.zeros((timesteps,output_size, 1))
        self.lin_out = np.zeros((timesteps,hidden_units, 1))
        self.Input = np.zeros((timesteps, 1))
        self.hidden_units = hidden_units
        self.timesteps = timesteps
        self.output_size = output_size
        self.state = np.zeros((timesteps, hidden_units,1))
        self.bptt_unfold = bptt_unfold
        self.type = 'rnn'
    def input_size(self):
        return self.timesteps

    def output_size(self):
        return self.output_size

    def forward(self, Input):
        self.state = np.zeros((self.timesteps, self.hidden_units,1))
        for step in range(self.timesteps):
            self.Input = np.zeros(Input.shape)
            self.Input[step] = Input[step]
            # act1 = np.expand_dims(np.dot(self.inp_weights, self.Input), axis=1)
            act1 = np.dot(self.inp_weights, self.Input)
            act2 = np.dot(self.recurrent_weights, self.state[step])
            self.lin_out[step] = act1 + act2
            self.state[step] = tanh(self.lin_out[step])
            self.out[step] = np.dot(self.out_weights, self.state[step])
        return self.out[self.timesteps-1]

    def backward(self, dAct, Input):
        
        d_inp_weights = np.zeros(self.inp_weights.shape)
        d_recurrent_weights = np.zeros(self.recurrent_weights.shape)
        d_out_weights = np.zeros(self.out_weights.shape)
        
        d_inp_weights_i = np.zeros(self.inp_weights.shape)
        d_recurrent_weights_i = np.zeros(self.recurrent_weights.shape)
        
        for step in range(self.timesteps):
            self.Input = np.zeros(Input.shape)
            self.Input[step] = Input[step]
            d_out_weights_i = np.dot(dAct, self.state[step].T)
            dAct__d_state_i = np.dot(self.out_weights.T, dAct)
            d_state_i__d_lin_out_i = 1.0 - np.square(tanh(self.lin_out[step]))
            #Check this shape!!!!
            d_Act__d_lin_out_i = np.dot(dAct__d_state_i.T , d_state_i__d_lin_out_i)
            #print(np.shape(d_Act__d_lin_out_i))
            for bck_thr_t in range(step, max(step-self.bptt_unfold, 1),-1):
                d_inp_weights_ii = np.dot(d_Act__d_lin_out_i, self.Input.T)
                d_recurrent_weights_ii = np.dot(d_Act__d_lin_out_i, self.state[step-1].T)
                
                d_inp_weights_i += d_inp_weights_ii
                d_recurrent_weights_i += d_recurrent_weights_ii
            
            d_inp_weights += d_inp_weights_i
            d_recurrent_weights += d_recurrent_weights_i
            d_out_weights += d_out_weights_i
            
        return d_inp_weights, d_recurrent_weights, d_out_weights


class RNN:
    def __init__(self, timesteps, output_size, hidden_units, bptt_unfold):
        self.inp_weights = np.random.randn(hidden_units, timesteps)*0.01
        self.recurrent_weights = np.random.randn(hidden_units, hidden_units)*0.01
        self.out_weights = np.random.randn(output_size, hidden_units)*0.01
        self.out = np.zeros((timesteps,output_size, 1))
        self.lin_out = np.zeros((timesteps,hidden_units, 1))
        self.Input = np.zeros((timesteps, 1))
        self.hidden_units = hidden_units
        self.timesteps = timesteps
        self.output_size = output_size
        self.state = np.zeros((timesteps, hidden_units,1))
        self.bptt_unfold = bptt_unfold
        self.type = 'rnn'
    def input_size(self):
        return self.timesteps

    def output_size(self):
        return self.output_size

    def forward(self, Input):
        self.state = np.zeros((self.timesteps, self.hidden_units,1))
        for step in range(self.timesteps):
            self.Input = np.zeros(Input.shape)
            self.Input[step] = Input[step]
            # act1 = np.expand_dims(np.dot(self.inp_weights, self.Input), axis=1)
            act1 = np.dot(self.inp_weights, self.Input)
            act2 = np.dot(self.recurrent_weights, self.state[step])
            self.lin_out[step] = act1 + act2
            self.state[step] = tanh(self.lin_out[step])
            self.out[step] = np.dot(self.out_weights, self.state[step])
        return self.out[self.timesteps-1]

    def backward(self, dAct, Input):
        
        d_inp_weights = np.zeros(self.inp_weights.shape)
        d_recurrent_weights = np.zeros(self.recurrent_weights.shape)
        d_out_weights = np.zeros(self.out_weights.shape)
        
        d_inp_weights_i = np.zeros(self.inp_weights.shape)
        d_recurrent_weights_i = np.zeros(self.recurrent_weights.shape)
        
        for step in range(self.timesteps):
            self.Input = np.zeros(Input.shape)
            self.Input[step] = Input[step]
            d_out_weights_i = np.dot(dAct, self.state[step].T)
            dAct__d_state_i = np.dot(self.out_weights.T, dAct)
            d_state_i__d_lin_out_i = 1.0 - np.square(tanh(self.lin_out[step]))
            #Check this shape!!!!
            d_Act__d_lin_out_i = np.dot(dAct__d_state_i.T , d_state_i__d_lin_out_i)
            #print(np.shape(d_Act__d_lin_out_i))
            for bck_thr_t in range(step, max(step-self.bptt_unfold, 1),-1):
                d_inp_weights_ii = np.dot(d_Act__d_lin_out_i, self.Input.T)
                d_recurrent_weights_ii = np.dot(d_Act__d_lin_out_i, self.state[step-1].T)
                
                d_inp_weights_i += d_inp_weights_ii
                d_recurrent_weights_i += d_recurrent_weights_ii
            
            d_inp_weights += d_inp_weights_i
            d_recurrent_weights += d_recurrent_weights_i
            d_out_weights += d_out_weights_i
            
        return d_inp_weights, d_recurrent_weights, d_out_weights


class DenseLayer:
    def __init__(self, input_size, output_size, activation_fnc='sigmoid'):
        self.weights = np.random.randn(output_size, input_size)*0.01
        self.bias = np.zeros((output_size, 1))
        self.act_fnc = activation_fnc
        self.out = np.zeros((output_size, 1))
        self.lin_out = np.zeros((output_size, 1))
        self.Input = np.zeros((input_size, 1))
        self.input_size = input_size
        self.output_size = output_size
        self.type = 'dense'
    def input_size(self):
        return self.input_size

    def output_size(self):
        return self.output_size

    def forward(self, Input):
        self.Input = Input
        act = np.dot(self.weights, self.Input) + self.bias
        self.lin_out = act
        if self.act_fnc == 'sigmoid':
            self.out = sigmoid(act)
            return self.out
        elif self.act_fnc == 'relu':
            self.out = relu(act)
            return self.out
        elif self.act_fnc == 'linear':
            self.out = act
            return self.out

    def backward(self, dAct):
        if self.act_fnc == 'sigmoid':
            dLin = dAct*sigmoid(self.lin_out)*(1-sigmoid(self.lin_out))
        elif self.act_fnc == 'linear':
            dLin = dAct

        mult = self.Input.shape[1]

        dWeight = (1/mult)*np.dot(dLin, self.Input.T)
        dBias = (1/mult)*np.expand_dims(np.sum(dLin, axis=1), axis=1)
        dActPrev = np.dot(self.weights.T, dLin)

        return dWeight, dBias, dActPrev





