import numpy as np
import random

class Layer:
    """
    Base class for layers in the network.

    Arguments:

        `inbound_layers`: A list of layers with edges into this layer.
    """
    def __init__(self, incoming_layers=[]):
        """ Simple constructor """
        self.incoming_layers = incoming_layers
        self.value = None
        self.outbound_layers = []
        self.gradient = 0.
        
        for layer in incoming_layers:
            layer.outbound_layers.append(self)

    def forward():
        """
        Every layer that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward():
        """
        Every layer that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError

    def g_prime(self):
        if isinstance(self.value, (np.ndarray, np.generic)):
            #return self.value.dot(( 1 - self.value))
            return np.multiply(self.value, (1-self.value))
        return self.value * (1 - self.value)


class Input(Layer):
    """
    A generic input into the network.
    """
    def __init__(self):
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        pass

class LogisticRegression(Layer):
    """ Represets all the Hidden Layers """
    def __init__(self, inbound_layer, weights, bias):
        Layer.__init__(self, [inbound_layer, weights, bias])
        
    def forward(self):
        inputs = self.incoming_layers[0].value
        weights = self.incoming_layers[1].value
        bias = self.incoming_layers[2].value
        h = np.sum(np.dot(inputs.T, weights))
        self.value = h + bias

    def backward(self):
        """ Calculates the gradient based on the output values."""
        incoming_gradient = self.outbound_layers[0].gradient
        start = True
        prev_layers = self.incoming_layers[:0:-1]
        for i in range(len(prev_layers)):
            n = prev_layers[i]
            if not start:
                incoming_gradient = prev_layers[i-1].gradient
            start = False
            h = n.value.T * incoming_gradient
            n.gradient +=  np.dot(h, n.g_prime())
            
    def query(self, x):
        weights = self.incoming_layers[1].value
        bias = self.incoming_layers[2].value
        orig_value = np.copy(self.value)
        
        h = np.dot(x, weights)
        self.value = h + bias
        self.outbound_layers[0].forward()
        self.value = np.copy(orig_value)
        return self.outbound_layers[0].value
            

class Sigmoid(Layer):
    """ Represents a layer that performs the sigmoid activation function. """
    
    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        Calculate Sigmoid

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.incoming_layers[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.incoming_layers}
        # Sum the partial with respect to the input over all the outputs.
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.incoming_layers[0]] += sigmoid * (1 - sigmoid) * grad_cost


class Softmax(Layer):
    """ Represents a layer that performs the sigmoid activation function. """
    
    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _softmax(self, x):
        """
        Calculate Sigmoid

        `x`: A numpy array-like object.
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward(self):
        """ Perform the sigmoid function and set the value."""

        input_value = self.incoming_layers[0].value
        self.value = self._softmax(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the softmax function.
        """
        if isinstance(self.gradient, float):
            self.gradient = np.zeros_like(self.value)

        for n in self.outbound_layers:
            incoming_grad = n.gradient
        self.gradient += incoming_grad * self.g_prime()

class CrossEntropy(Layer):
    def __init__(self, inbound_layer):
        """
        The multi-class classifier.
        Should be used as the last layer for a network.

        Arguments
            `inbound_layer`: A layer with an activation function.
        """
        # Call the base class' constructor.
        Layer.__init__(self, [inbound_layer])
        self.ideal_output = None
        self.n_inputs = None
        self.value = 0.

    def forward(self):
        """
        Calculates the cross entropy.
        """
        # Save the computed output for backward.
        self.computed_output = self.incoming_layers[0].value
        self.value += -np.sum(np.multiply(self.ideal_output, np.log(self.computed_output)))

    def backward(self):
        """
        Calculates the gradient of the cost.
        """
        self.gradient += (self.computed_output - self.ideal_output)

            
class MSE(Layer):
    def __init__(self, inbound_layer):
        """
        The mean squared error cost function.
        Should be used as the last layer for a network.

        Arguments:
            `inbound_layer`: A layer with an activation function.
        """
        # Call the base class' constructor.
        Layer.__init__(self, [inbound_layer])
        """
        These two properties are set during topological_sort()
        """
        # The ideal_output for forward().
        self.ideal_output = None
        # The number of inputs for forward().
        self.n_inputs = None

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # Save the computed output for backward.
        self.computed_output = self.incoming_layers[0].value
        first_term = 1. / (2. * self.n_inputs)
        norm = np.linalg.norm(self.ideal_output - self.computed_output)
        self.value = first_term * np.square(norm)

    def backward(self):
        """
        Calculates the gradient of the cost.
        """
        self.gradients[self.incoming_layers[0]] = -2 * (self.ideal_output - self.computed_output)


def topological_sort(feed_dict, ideal_output):
    """
    Sort the layers in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
    `ideal_output`: The correct output value for the last activation layer.

    Returns a list of sorted layers.
    """

    input_layers = [n for n in feed_dict.keys()]

    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]
        if isinstance(n, CrossEntropy) or isinstance(n, MSE):
            n.ideal_output = ideal_output
            # there is only 1 input in this example
            n.n_inputs = 1

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L
