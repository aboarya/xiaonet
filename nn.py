import xiaoloader


def train_mnist():
    training_data = xiaoloader.load_mnist_training()
    for index, data in training_data.iteritems():
        digit = data['digit']
        img = data['img']

        
