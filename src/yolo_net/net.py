
class Net(object):
    '''
    Base Net class
    '''
    def __init__(self):
        '''
        构建网络
        '''
        pass

    def _conv2d_layer(self,name):
        pass

    def _max_pooling_layer(self,name):
        pass

    def _connection_layer(self,name):
        pass

    def _variable_wights(self,shape):
        pass

    def _variable_biases(self,shape):
        pass

    def test(self):
        return NotImplementedError

    def train(self):
        return NotImplementedError

    def loss(self):
        return NotImplementedError

