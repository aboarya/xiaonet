import numpy as np

import xiaoloader
from xiaonet import *

def train_mnist():
    mnist_data = xiaoloader.load_mnist_training()
    training_data = {k:v for k,v in mnist_data.items() if int(k) >= (len(mnist_data.keys())*.7)}
    validation_data = {k:v for k,v in mnist_data.items() if int(k) < (len(mnist_data.keys())*.7)}
    
    inputs, weights, bias = Input(), Input(), Input()
    f = HiddenLayers(inputs, weights, bias)
    g = Softmax(f)
    cost = CrossEntropy(g)

    print("beginning training")
    for index, data in training_data.items():
        digit = data['digit']
        
        x =  data['img']
        w = np.random.normal(0.0, pow(10, -0.5),
                             (x.shape[0], 10))
        
        b = np.random.normal(0.0, pow(10, -0.5),
                             10)

        ideal_output = data['label']
        
       

        feed_dict = {inputs: x, weights: w, bias: b}
        loss = train_SGD(feed_dict, ideal_output, [weights, bias], 1000)

    print("beginning validation")
    final = []
    for index, data in validation_data.items():
        digit = data['digit']
        x =  data['img']
        values = f.query(x)
        #print(np.amax(values))
        final.append([digit, np.argmax(values)])

    correct = 0.
    for combo in final:
        if combo[0] == combo[1]:
            correct += 1
    print(correct/len(final))
train_mnist()
