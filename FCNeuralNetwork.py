import numpy as np

class Layer:
    def __init__(self):
        self.input = None;
        self.output = None;
    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        pass;
    # computes dE/dX for a given dE/dY and updates any parameters
    def backward_propagation(self, output_error, learning_rate):
        pass;
    
class FCLayer(Layer):
    # input_size = number of input neurons;
    # output_size = number of output neurons;
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5; #inputsize x outputsize matrix
        self.bias = np.random.rand(1, output_size); #1xoutputsize matrix
        
    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data;
        self.output = np.dot(self.input, self.weights) + self.bias;
        return self.output;
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T);
        weights_error = np.dot(self.input.T, output_error);
        #dBias = output_error;
        
        # update parameters
        self.weights -= learning_rate * weights_error;
        self.bias -= learning_rate * output_error;
        return input_error;
    
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation;
        self.activation_prime = activation_prime;
        
    # returns the activated output
    def forward_propagation(self, input_data):
        self.input = input_data;
        self.output = self.activation(self.input);
        return self.output;
    
    # returns dE/dX for a given dE/dY
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error;

######################################################

# define some activation functions
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x) ** 2;

def sigmoid(x):
    return 1 / (1 + np.e**(-x));

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x));

def ReLU(x):
    if (x < 0):
        return 0;
    else:
        return x;

def ReLU_prime(x):
    if (x < 0):
        return 0;
    else:
        return 1;
    
# mean squared error (loss function)
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true)/y_true.size;

######################################################

class Network:
    def __init__(self):
        self.layers = [];
        self.loss = None;
        self.loss_prime = None;
        
    def add(self, layer):
        self.layers.append(layer);
        
    def use(self, loss, loss_prime):
        self.loss = loss;
        self.loss_prime = loss_prime;
    
    def predict(self, input_data):
        samples = len(input_data);
        result = [];
        
        for i in range(samples):
            output = input_data[i];
            for layer in self.layers:
                output = layer.forward_propagation(output);
            result.append(output);
            
        return result;
        
    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train);
        
        for i in range(epochs):
            err = 0;
            for j in range(samples):
                output = x_train[j];
                for layer in self.layers:
                    output = layer.forward_propagation(output);

                err += self.loss(y_train[j], output);

                error = self.loss_prime(y_train[j], output);
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate);

            err /= samples;
            print('epoch %d/%d   error=%f' % (i+1, epochs, err));