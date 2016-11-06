import numpy as np

import xiaoloader
from xiaonet import *


def train_SGD(feed_dict, ideal_output, trainables=[], epochs=1, learning_rate=0.1):
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
    
    # Forward pass
    for n in sorted_layers[:-1]:
        n.forward()
        
    # Forward again
    #for n in sorted_layers[:-1]:
    #    n.forward()

    # Ouput
    output_layer = sorted_layers[-1]
    output_layer.forward()
    
    # Backward pass
    reversed_layers = sorted_layers[::-1] # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    
    for n in reversed_layers:
        n.backward()

    # Performs SGD
    # Get a list of the partials with respect to each trainable input.
    partials = [n.gradient for n in trainables]
    # Loop over the trainables
    for n in range(len(trainables)):
        # Change the trainable's value by subtracting the learning rate
        # multiplied by the partial of the cost with respect to this
        # trainable.
        #print(partials[n])
        trainables[n].value -= learning_rate * partials[n]

    return sorted_layers[-1].value

validation_data = None
inputs, weights, bias = Input(), Input(), Input()
f = HiddenLayers(inputs, weights, bias)
g = Softmax(f)
cost = CrossEntropy(g)

mnist_data = xiaoloader.load_mnist_training()
training_data = {k:v for k,v in mnist_data.items() if int(k) >= (len(mnist_data.keys())*.7)}
validation_data = {k:v for k,v in mnist_data.items() if int(k) < (len(mnist_data.keys())*.7)}

def train_mnist():
    print("beginning training")
    i = 0
    for index, data in training_data.items():
        digit = data['digit']
        
        x =  data['img']
        w = np.random.normal(0.0, pow(10, -0.5), 784)
        
        b = np.random.normal(0.0, pow(10, -0.5), 10)
     
        ideal_output = data['label']
        feed_dict = {inputs: x, weights: w, bias: b}
        loss = train_SGD(feed_dict, ideal_output, [weights, bias], 1000)
        if i > 5: break
        i+= 1
    return loss

for i in range(2):
    print('Epoch: ' + str(i) + ', Loss: ' + str(train_mnist())) 
    
#import sys;sys.exit(1)
correct = 0
for index, data in validation_data.items():
    digit = data['digit']
    x =  data['img']
    values = f.query(x)
    if digit == np.argmax(values):
        print(">>>>>>??????????????????????")
        print(digit, values)
        correct += 1
    break
print("accuracy: ", (correct/len(validation_data.keys())))
