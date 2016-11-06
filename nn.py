import numpy as np

import xiaoloader
from xiaonet import *

def train_mnist():
    training_data = xiaoloader.load_mnist_training()
    for index, data in training_data.items():
        digit = data['digit']
        
        inputs, weights, bias = Input(), Input(), Input()
        
        x =  data['img']
        w = np.random.normal(0.0, pow(10, -0.5),
                             (x.shape[0], 10))
        
        b = np.random.normal(0.0, pow(10, -0.5),
                             10)

        ideal_output = data['label']
        
        f = Hidden(inputs, weights, bias)
        g = Softmax(f)
        cost = CrossEntropy(g)

        feed_dict = {inputs: x, weights: w, bias: b}

        train_SGD(feed_dict, ideal_output, [weights, bias], 1000)
        break

train_mnist()
