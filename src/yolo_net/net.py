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
    _weights_decay = 0 # 权值衰减
    _trainable = False

    def __init__(self):
        '''
        构建网络
        '''
        pass

    def _conv2d_layer(self,name,input,kernel_size,output_size,stride,activate_fc='leaky'):
        '''
        卷积层，采用leaky作为激活函数
        :param name:
        :param input:
        :param kernel_size:
        :param output_size:
        :param stride:
        :param activate_fc:
        :return:
        '''
        with tf.name_scope(name):
            input_size = tf.shape(input)[3]

            wights = self._variable_weights([kernel_size,input_size,output_size])
            biases = self._variable_biases(output_size)
            conv = tf.nn.conv2d(input,wights,strides=[1,stride,stride,1],padding='SAME')
            layer_output = tf.add(conv,biases)

            return self._activate_func(layer_output,activate_fc)

    def _activate_func(self,x,name):
        '''
        激活函数leaky，见yolo_v1论文
        if x >0:
            x=x
        else
            x= alpha*x
        '''

        if name == 'leaky':
            temp_x = tf.cast(x>0,tf.float32)
            out_put = tf.add(tf.multiply(temp_x,x),tf.multiply(1-temp_x,x)*self._weights_decay)
        elif name == 'linear':
            out_put = x
        else:
            out_put = x

        return out_put

    def _max_pooling_layer(self,name,input,kernel_size,stride):
        '''
        最大池化层，采用边缘检测
        :param name:
        :param input:
        :param kernel_size:
        :param stride:
        :return:
        '''
        with tf.name_scope(name):
            return tf.nn.max_pool(input,[1,kernel_size,kernel_size,1],strides=stride,padding='SAME')

    def _fc_layer(self,name,input,input_size,output_size,activate_fc):
        '''
        全连接层
        :param name:
        :return:
        '''

        with tf.name_scope(name):
            weights = self._variable_weights([input_size,output_size])
            biases = self._variable_biases(output_size)
            layer_output = self._activate_func(tf.add(tf.matmul(input,weights),biases),activate_fc)
            return layer_output

    def _variable_weights(self,shape):
        '''

        如果进行训练，则需要加上权值衰减
        :param shape: if conv
                            shape = [kernel_size,kernel_size,channels,filters]
                        if connection
                            shape = [input_size,output_size]
        :return:
        '''
        weights = tf.Variable(tf.truncated_normal(shape,stddev=0.1),
                              dtype=tf.float32)
        if self._trainable:
            weights_decay = tf.multiply(weights,self._weights_decay)
            tf.add_to_collection('losses',weights_decay)
        return weights

    def _variable_biases(self,shape):
        '''
        bias初始赋值为0.1
        :param shape:
        :return:
        '''
        return tf.Variable(tf.constant(0.1,shape=shape),dtype=tf.float32)

    def test(self):
        return NotImplementedError

    def train(self):
        return NotImplementedError

    def loss(self):
        return NotImplementedError

