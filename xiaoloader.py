import csv
import urllib.request
import codecs
import numpy as np

MNIST_training_csv_url = "https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv"

def load_mnist_training():
    """ Returns 100 MNIST samples """
    data = dict()
    ftpstream = urllib.request.urlopen(MNIST_training_csv_url)
    rows = csv.reader(codecs.iterdecode(ftpstream, 'utf-8'))
    i = 0
    for row in rows:
        data[str(i)] = dict(
            digit=int(row[0]),
            img=(np.asfarray(row[1:]) / 255.0 * 0.99) + 0.01,
            
        )
        label = np.zeros(10) + 0.01
        label[int(row[0])] = 0.99
        data[str(i)]['label'] = label
        
        i+=1
    return data
    
