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

    def linear_transform(self):
        """ perform linear transorm """
        h = np.sum(np.dot(self.x.T, self.w))
        self.value = h + self.b


class Input(Layer):
    """ Represets all the Hidden Layers """
    def __init__(self, feature, weights, bias):
        Layer.__init__(self)
        self.x = feature
        self.w = weights
        self.b = bias
        
    def forward(self):
        Layer.linear_transform(self)

    def backward(self):
        """ Calculates the gradient based on the output values."""
        #self.input_gradient = self.outbound_layers[0].gradient
        self.d_w1 = self.outbound_layers[0].gradient * self.x.T
        self.d_b1 = self.outbound_layers[0].gradient
        self.d_b2 = self.outbound_layers[0].gradient1
        

                                         
class Linear(Layer):
        
    def __init__(self, input_layer, weights, bias):
        Layer.__init__(self, [input_layer])
        self.x = self.incoming_layers[0].value
        self.w = weights
        self.b = bias

    def forward(self):
        Layer.linear_transform(self)

    def backward(self):
        #self.d_l3 = self.d_b2 = self.outbound_layers[0].gradient
        self.gradient = self.input_gradient * self.w.T
        self.gradient1 = self.outbound_layers[0].gradient
        #self.d_w2 =  self.d_l * self.x.T
        #self.gradient = self.input_gradient + self.d_l3 + self.d_l2 + self.d_w2 + self.d_l3
        pass
        

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
        """ Perform the sigmoid function and set the value. """
        input_value = self.incoming_layers[0].value
        self.value = self._softmax(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the softmax function.
        """
        # gradient of cross entropy which is 1
        self.input_gradient = self.outbound_layers[0].gradient
        self.gradient = self.input_gradient * (self.value - self.ideal_output)

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
        self.value = 1.

    def forward(self):
        """
        Calculates the cross entropy.
        """
        # Save the computed output for backward.
        self.computed_output = self.incoming_layers[0].value
        self.value = -np.sum(np.multiply(self.ideal_output, np.log(self.computed_output)))

    def backward(self):
        """
        Calculates the gradient of the cost.
        """
        self.gradient = 1


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
        if isinstance(n, CrossEntropy) or isinstance(n, Softmax):
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
