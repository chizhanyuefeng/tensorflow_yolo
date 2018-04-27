import tensorflow as tf


class Net(object):
    '''
    Base Net class
    '''

    _batch_size = 0
    _image_size = 0
    _momentum = 0
    _decay = 0
    _learning_rate = 0
    _max_objects = 0

    def __init__(self):
        '''
        构建网络
        '''
        pass

    def _conv2d_layer(self,name,input,kernel_size,output_size,stride,):
        with tf.name_scope(name):
            input_size = tf.shape(input)[3]

            wights = self._variable_wights(kernel_size,input_size,output_size)
            biases = self._variable_biases(output_size)
            conv = tf.nn.conv2d(input,wights,strides=[1,stride,stride,1],padding='SAME')
            layer_output = tf.add(conv,biases)

            return self._activation_func(layer_output)

    def _activation_func(self,x):
        '''
        激活函数，见yolo_v1论文
        if x >0:
            x=x
        else
            x= alpha*x
        '''
        temp_x = tf.cast(x>0,tf.float32)

        return tf.add(tf.multiply(temp_x,x),tf.multiply(1-temp_x,x)*self._alpha)


    def _max_pooling_layer(self,name):
        pass

    def _connection_layer(self,name):
        pass

    def _variable_wights(self,kernel_size,inpout_size,output_size):
        pass

    def _variable_biases(self,shape):
        pass

    def test(self):
        return NotImplementedError

    def train(self):
        return NotImplementedError

    def loss(self):
        return NotImplementedError

