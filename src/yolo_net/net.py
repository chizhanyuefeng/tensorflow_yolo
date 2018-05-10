
import time
import cv2
import numpy as np
import tensorflow as tf
import utils.cfg_file_parser as cp

from tensorflow.python import pywrap_tensorflow

WIGHTS_NAME = 0

class Net(object):
    '''
    Base Net class
    '''

    _momentum = 0.9
    _learning_rate = 0
    _max_objects_per_image = 20
    _weights_decay = 0.0005 # 权值衰减
    _trainable = False
    _cfg_file_path = ''
    _leaky_alpha = 0.1
    _model_path = ''
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __init__(self):
        '''
        构建网络
        '''
        self._sess = None

    def __del__(self):
        if self._sess!=None:
            self._sess.close()

    def __get_session(self):
        if self._sess==None:
            self._sess = tf.Session()
        return self._sess

    def _conv2d_layer(self,name,input,kernel_size,channels,filters,stride,activate_fc='leaky'):
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
        if WIGHTS_NAME:
            with tf.name_scope(name):
                wights = self._variable_weights([kernel_size,kernel_size,channels,filters])
                biases = self._variable_biases([filters])
                conv = tf.nn.conv2d(input,wights,strides=[1,stride,stride,1],padding='SAME')
                layer_output = tf.add(conv,biases)
        else:
            wights = self._variable_weights([kernel_size, kernel_size, channels, filters])
            biases = self._variable_biases([filters])
            conv = tf.nn.conv2d(input, wights, strides=[1, stride, stride, 1], padding='SAME')
            layer_output = tf.add(conv, biases)

        return self._activate_func(layer_output, activate_fc)

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
            out_put = tf.add(tf.multiply(temp_x,x),tf.multiply(1-temp_x,x)*self._leaky_alpha)
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
        if WIGHTS_NAME:
            with tf.name_scope(name):
                weights = self._variable_weights([input_size,output_size])
                biases = self._variable_biases([output_size])
                # 如果是最后一层的卷积输出需要进行reshape
                input_shape = input.get_shape().as_list()

                if len(input_shape)>2:
                    dim = 1
                    for i in range(1,len(input_shape)):
                        dim = dim*input_shape[i]
                    input = tf.transpose(input,[0,3,1,2])
                    # 这里reshape时候，需要使用-1，而不能使用None
                    input = tf.reshape(input,(-1,dim))
                layer_output = self._activate_func(tf.add(tf.matmul(input,weights),biases),activate_fc)
        else:
            weights = self._variable_weights([input_size, output_size])
            biases = self._variable_biases([output_size])
            # 如果是最后一层的卷积输出需要进行reshape
            input_shape = input.get_shape().as_list()

            if len(input_shape) > 2:
                dim = 1
                for i in range(1, len(input_shape)):
                    dim = dim * input_shape[i]
                input = tf.transpose(input, [0, 3, 1, 2])
                # 这里reshape时候，需要使用-1，而不能使用None
                input = tf.reshape(input, (-1, dim))
            layer_output = self._activate_func(tf.add(tf.matmul(input, weights), biases), activate_fc)

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

    def construct_graph(self):
        '''
        构建网络tensor graph
        :return:
        '''

        net_params, train_params, net_structure_params =cp.parser_cfg_file(self._cfg_file_path)

        self._batch_size = int(net_params['batch_size'])
        self._image_size = int(net_params['image_size'])
        self._classes_num = int(net_params['classes_num'])
        self._cell_size = int(net_params['cell_size'])
        self._boxes_per_cell = int(net_params['boxes_per_cell'])
        self._iou_threshold = float(net_params['iou_threshold'])
        self._score_threshold = float(net_params['score_threshold'])

        self._image_input = tf.placeholder(tf.float32, shape=[None,self._image_size,self._image_size,3 ])
        print('开始构建yolo网络...')
        self._net_output = self._image_input
        for i in range(len(net_structure_params)):
            name = net_structure_params[i]['name']
            if 'convolutional' in name:
                filters = int(net_structure_params[i]['filters'])
                size = int(net_structure_params[i]['size'])
                stride = int(net_structure_params[i]['stride'])
                pad = int(net_structure_params[i]['pad'])
                #channels = int(net_structure_params[i]['channels'])
                channels = self._net_output.get_shape().as_list()[3]
                activation = net_structure_params[i]['activation']
                self._net_output = self._conv2d_layer(name,self._net_output,size,channels,filters,stride,activation)
                print('建立[%s]层，卷积核大小=[%d],个数=[%d],步长=[%d],激活函数=[%s]'%
                      (name,size,filters,stride,activation))
            elif 'maxpool' in name:
                size = int(net_structure_params[i]['size'])
                stride = int(net_structure_params[i]['stride'])
                self._net_output = self._max_pooling_layer(name,self._net_output,size,[1,stride,stride,1])
                print('建立[%s]层，pooling大小=[%d],步长=[%d]' %
                      (name, size,stride))
            elif 'connected' in name:
                shape = self._net_output.get_shape().as_list()
                input_size = 1
                for j in range(1,len(shape)):
                    input_size = input_size* shape[j]
                output_size = int(net_structure_params[i]['output'])
                activation = net_structure_params[i]['activation']
                self._net_output = self._fc_layer(name,self._net_output,input_size,output_size,activation)
                print('建立[%s]层，输入层=[%d],输出层=[%d],激活函数=[%s]'%
                      (name, input_size,output_size, activation))
            else:
                print('网络配置文件出错！')
                break
        print('构建完网络结构！')

    def load_model(self):

        # 用来打印model的变量名字和数据
        # reader = pywrap_tensorflow.NewCheckpointReader(model_file)
        # var_to_shape_map = reader.get_variable_to_shape_map()
        # for key in var_to_shape_map:
        #     print("tensor_name: ", key)
        #     print(reader.get_tensor(key))

        #self.__get_session().run(tf.global_variables_initializer())
        self._model_saver = tf.train.Saver()
        self._model_saver.restore(self.__get_session(),self._model_path)

    def test(self,image_path):
        '''
        测试图片
        :param image_path:
        :return:
        '''
        start = time.time()
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image,(self._image_size,self._image_size)).reshape([1,self._image_size,self._image_size,3])
        normaliztion_image = resized_image/255*2 -1
        output = self.__get_session().run(self._net_output,feed_dict={self._image_input:normaliztion_image})
        during = str(time.time() - start)
        print('耗时=',during,',输出结果为：',output)
        self.__interpert_output(self._net_output[0])

    def __interpert_output(self,output):
        '''
        解析网络输出
        :param output: shape = [1470,]
        :return:
        '''
        cell_size = self._classes_num+self._boxes_per_cell*1+4*2
        output = tf.reshape(output,[self._cell_size,self._cell_size,cell_size])

        # 获取每个类的概率值,置信度，bbox
        classes_probs = output[:,:,0:20]
        boxes_confidence = output[:,:,20:22]
        boxes = output[:,:,22:]

        # 将每个类的概率和置信度相乘得到scores
        confidence_scores = tf.zeros([self._cell_size,self._cell_size,2,20])
        for i in range(2):
            for j in range(20):
                confidence_scores[:,:,i,:j] = boxes_confidence[:,:,i]*classes_probs[:,:,:j]

        # 用于过滤数据,shape = [7,7,2,20]
        score_filter = tf.constant(confidence_scores>=self._score_threshold,dtype=tf.bool)
        




    def train(self):
        return NotImplementedError

    def loss(self):
        return NotImplementedError

