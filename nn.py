import numpy as np

import xiaoloader
from xiaonet import *


def train_SGD(feed_dict, ideal_output, trainables=[], epochs=1, learning_rate=0.5):
    """
    Performs many forward passes and a backward passes through
    a list of sorted Layers while performing stochastic gradient
    descent.

    Arguments:

        `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.
        `ideal_output`: The correct output value for the last activation layer.
        `trainables`: Inputs that need to be modified by SGD.
        `epochs`: The number of times to train against all training inputs.
        `learning_rate`: The step size for changes by each gradient.
    """

    sorted_layers = topological_sort(feed_dict, ideal_output)
    reversed_layers = sorted_layers[::-1] # see: https://docs.python.org/2.3/whatsnew/section-slices.html
            
    for i in range(epochs):
        # Forward pass
        for n in sorted_layers:
            n.forward()
            
        for n in reversed_layers:
            n.backward()

        # Performs SGD
        input_layer = sorted_layers[0]
        hidden_layer = input_layer.outbound_layers[0]
        partials = (input_layer.d_w2, input_layer.d_b1, input_layer.d_b2,)

        # Loop over the trainables
        for n in range(len(trainables)):
            trainables[n] -= learning_rate * partials[n]
        print("Loss is now : ", reversed_layers[0].value)

    return (reversed_layers[0].value,)+partials

validation_data = None

mnist_data = xiaoloader.load_mnist_training()
training_data = {k:v for k,v in mnist_data.items() if int(k) >= (len(mnist_data.keys())*.7)}
validation_data = {k:v for k,v in mnist_data.items() if int(k) < (len(mnist_data.keys())*.7)}

def train_mnist():
    print("beginning training")
    keys = sorted(training_data.keys())
    i = 0
    for key in keys:
        i += 1
        w1 = np.random.normal(0.0, pow(10, -0.5), (784,))
        w2 = np.random.normal(0.0, pow(10, -0.5), (784, 10))

        b1 = np.random.normal(0.0, pow(10, -0.5), 10)
        b2 = np.random.normal(0.0, pow(10, -0.5), 10)

        data = training_data[key]
        digit = data['digit']
        print(digit)
        x =  data['img']
        inputs = Input(x, w1, b1)
        f = Linear(inputs, w2, b2)
        g = Softmax(f)
        distance = CrossEntropy(g)
        
        ideal_output = data['label']
        feed_dict = {inputs: x}
        loss, w2, b1, b2 = train_SGD(feed_dict, ideal_output, [w2, b1, b2], 10)
        print("")
        if i > 5: break
    
train_mnist()    
import sys;sys.exit(1)
correct = 0
for index, data in validation_data.items():
    digit = data['digit']
    x =  data['img']
    values = f.query(x)
    if digit == np.argmax(values):
        print(digit, values)
        correct += 1
    break
print("accuracy: ", (correct/len(validation_data.keys())))
