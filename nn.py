import numpy as np

import xiaoloader
from xiaonet import *

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
