import csv
import urllib2

MNIST_training_csv_url = "https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv"

def load_mnist_training():
    """ Returns 100 MNIST samples """
    response = urllib2.urlopen(MNIST_training_csv_url)
    rows = csv.reader(response)
    data = dict()
    for i in range(len(rows)):
        data[str(i)] = dict(
            digit=row[0],
            img=np.asfarray(row[1:]).flatten()
        )
    return data
    
