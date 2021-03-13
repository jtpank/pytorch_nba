import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import timeit as TIMEIT

#SOURCES:
#http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
#https://adventuresinmachinelearning.com/wp-content/uploads/2017/07/An-introduction-to-neural-networks-for-beginners.pdf


#RELU activation function and derivative
def relu(x):
    return np.maximum(0,x)
def d_relu(x):
    if x<=0:
        return 0
    else:
        return 1
#SIGMOID activation function and derivative
def a(x):
    return 1/(1+np.exp(-x))
def d_a(x):
    return (a(x)*(1-a(x)))
#calc feed forward from matrices (*SLOW*)
def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        #output array for nodes in l+1 layer
        h = np.zeros((w[l].shape[0],))
        #loop through weight array rows
        for row in range(w[l].shape[0]):
            #sum inside activation function
            f_sum = 0
            #loop through weight array columns
            for col in range(w[l].shape[1]):
                f_sum += w[l][row][col]*node_in[col]
            #bias add
            f_sum += b[l][row]
            #use activation function to calculate i-th (row-th) output
            #h1, h2, ....
            h[row] = a(f_sum)
    return h
#calc feed forward from matrices with vectorization (*FAST*)
def matrix_feed_forward_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = relu(z)
    return h
def feedForwardTest():
    #weight matrices
    w1 = np.array([[0.1,0.1,0.1],[0.3,0.3,0.3],[0.6,0.6,0.6]])
    w2 = np.zeros((1,3))
    w3 = np.array([[0.5,0.4,0.2]])
    w2[0,:] = np.array([0.1,0.6,0.3])
    #bias vectors
    b0 = np.array([0,0,0])
    b1 = np.array([0.4,0.4,0.4])
    b2 = np.array([0.2, 0.2, 0.2])
    b3 = np.array([0.1,0.2,0.3])
    w = [w1,w2, w3]
    b = [b1,b2, b3]
    b00 = [b0,b0,b0]
    #dummy input vector
    x = [1.5,2.0,3.0]
    print(matrix_feed_forward_calc(4, x, w, b00))


#Setting up layers for neural net
#Batch inputs: _inputBatch = [[x1],[x2],...,[xn]]
#Batch weights: _weightBatch = [[w1],[w2],...,[wn]]
#for x1 is a vector of size M e.g. x1 = [i1, i2, ... ,im]
#for w1 is a vector of size M e.g. w1 = [j1, j2, ... ,jm]
#Bias vector is of size _N_Layers for N number of layers

_x1 = np.array([[2, 3.3, -1, 2],[1, 0.3, 1.1, -0.7]])
_w1 = np.array([[0.2,0.3,0.4,0.5],[0.1,0.05,0.8,-0.1]])
_w2 = np.array([0.1,-0.3,0.1,0.52])
_b1 = np.array([1,2,0.4,3])
_b2 = np.array([-1,-2,-0.4,-3])

#layer 1 output
#_l1_out = _w1.dot(_x1) + _b1
#layer 2 output
#_l2_out = _w2.dot(_l1_out) + _b2


#Can load a network's weights and biases
#OR 
#Can create a new network

class Layer_Dense:
    def __init__(self, num_inputs, num_neurons):
        #size of input coming in
        #how many neurons
        self.weights = 0.1*np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1,num_neurons))
    def forward(self, _INPUTS):
        self.output = np.dot(_INPUTS, self.weights) + self.biases
class RELU_Activation:
    def forward(self, _INPUTS):
        self.output = np.maximum(0,_INPUTS)

class SFTMAX_Activation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs
#softmax activation function implementation for final outputs:
# input (output layer of data) -> exponentiate input values (each value) -> normalize vector -> output
# S_ij = exp(z_ij) / sum_i(exp(z_ij))




l1 = Layer_Dense(4, 3)
act1 = RELU_Activation()

l2 = Layer_Dense(3, 2)
act2 = SFTMAX_Activation()

l1.forward(_x1)
act1.forward(l1.output)

l2.forward(act1.output)
act2.forward(l2.output)

print(act2.output)

def main():
    print("from_main")
    

if __name__ == "__main__":
    main()