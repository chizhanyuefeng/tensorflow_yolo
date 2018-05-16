
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

    _trainable = False
    _cfg_file_path = None

    _model_path = None
    _net_name = None

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
            out_put = tf.add(tf.multiply(temp_x,x),tf.multiply(1-temp_x,x)*self.__leaky_alpha)
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
            weights_decay = tf.multiply(weights,self.__weights_decay)
            tf.add_to_collection('losses',weights_decay)
        return weights

    def _variable_biases(self,shape):
        '''
        bias初始赋值为0.1
        :param shape:
        :return:
        '''
        return tf.Variable(tf.constant(0.1,shape=shape),dtype=tf.float32)


    def __load_train_params(self,train_params):
        '''
        初始化训练参数和构建tensor
        :param train_params:
        :return:
        '''
        self.__batch_size = int(train_params['batch_size'])
        self.__max_objects_per_image = int(train_params['max_objects_per_image'])
        self.__object_scale = float(train_params['object_scale'])
        self.__noobject_scale = float(train_params['noobject_scale'])
        self.__class_scale = float(train_params['class_scale'])
        self.__coord_scale = float(train_params['coord_scale'])
        self.__momentum = float(train_params['momentum'])
        self.__weights_decay = float(train_params['weights_decay'])
        self.__leaky_alpha = float(train_params['leaky_alpha'])
        self.__learning_rate = float(train_params['learning_rate'])
        self.__max_iterators = int(train_params['max_iterators'])

        # 输入标签，5 = [x,y,w,h,c]
        self.__labels = tf.placeholder(tf.float32,[self.__batch_size,self.__max_objects_per_image,5])

    def _construct_graph(self):
        '''
        构建网络tensor graph
        :return:
        '''

        net_params, train_params, net_structure_params =cp.parser_cfg_file(self._cfg_file_path)

        # 加载网络基本配置参数
        self._input_size = int(net_params['input_size'])
        self._classes_num = int(net_params['classes_num'])
        self._cell_size = int(net_params['cell_size'])
        self._boxes_per_cell = int(net_params['boxes_per_cell'])
        self._iou_threshold = float(net_params['iou_threshold'])
        self._score_threshold = float(net_params['score_threshold'])
        self.__leaky_alpha = float(net_params['leaky_alpha'])

        # 如需进行训练，则加载训练参数
        if self._trainable:
            self._image_input_tensor = tf.placeholder(tf.float32, shape=[self.__batch_size, self._input_size, self._input_size, 3])
            self.__load_train_params(train_params)
        else:
            self._image_input_tensor = tf.placeholder(tf.float32, shape=[None,self._input_size,self._input_size,3])

        print('开始构建yolo网络...')
        self._net_output = self._image_input_tensor

        # 从网络结构配置加载参数，进而构建网络
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
                print('第[%d]层：[%s]层，卷积核大小=[%d],个数=[%d],步长=[%d],激活函数=[%s]'%
                      (i,name,size,filters,stride,activation))

            elif 'maxpool' in name:
                size = int(net_structure_params[i]['size'])
                stride = int(net_structure_params[i]['stride'])
                self._net_output = self._max_pooling_layer(name,self._net_output,size,[1,stride,stride,1])
                print('第[%d]层：[%s]层，pooling大小=[%d],步长=[%d]' %
                      (i,name, size,stride))

            elif 'connected' in name:
                shape = self._net_output.get_shape().as_list()
                input_size = 1
                for j in range(1,len(shape)):
                    input_size = input_size* shape[j]
                output_size = int(net_structure_params[i]['output'])
                activation = net_structure_params[i]['activation']
                self._net_output = self._fc_layer(name,self._net_output,input_size,output_size,activation)
                print('第[%d]层：[%s]层，输入层=[%d],输出层=[%d],激活函数=[%s]'%
                      (i,name, input_size,output_size, activation))
            else:
                print(self._cfg_file_path,'网络配置文件出错！')
                return
        print('构建完网络结构！')

    def _load_model(self):
        '''
        加载模型
        :return:
        '''
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

        # 读取图片
        start_time = time.time()
        self._image = cv2.imread(image_path)
        self._image_height,self._image_width,_ = self._image.shape

        # 将图片resize成[448,448,3]，因为opencv读取图片后存储格式为BGR，所以需要再转为RGB
        resized_image = cv2.resize(self._image,(self._input_size,self._input_size))
        img_RGB = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)

        # 将数据进行归一化处理
        normaliztion_image = img_resized_np/255.0*2.0 -1
        input = np.zeros([1,448,448,3],np.float32)
        input[0] = normaliztion_image
        self._image_inputs = input

        # 加载模型
        self._load_model()

        # 将网络输出进行解析
        result = self.__interpert_output(self._net_output[0])
        during = str(time.time() - start_time)
        print('检测耗时=', during)

        if not result:
            return
        else:
            result_classes, result_bboxes, result_scores =result

        # 展示结果
        self.__show_result(result_classes, result_bboxes, result_scores)

    def __interpert_output(self,output):
        '''
        解析网络输出
        :param output: shape = [1470,]
        :return:
        '''

        # 获取每个类的概率值,置信度,bbox
        classes_probs = tf.reshape(output[0:7*7*20],[7,7,20])
        confidences = tf.reshape(output[7*7*20:7*7*22],[7,7,2])
        boxes = tf.reshape(output[7*7*22:],[7,7,2,4])

        # 将每个类的概率和置信度相乘得到scores
        confidence_scores_list = []
        for i in range(2):
            for j in range(20):
                temp = tf.multiply(confidences[:, :, i], classes_probs[:, :, j])
                confidence_scores_list.append(temp)

        # 得分过滤器,shape = [7,7,2,20]，找到大于阈值的得分，并保存其所在位置。
        confidence_scores = tf.reshape(tf.transpose(tf.stack(confidence_scores_list),[1,2,0]),[7,7,2,20])
        score_filter = confidence_scores>=self._score_threshold
        # filter 的shape = [N,4]，其中N为多少个符合阈值的数据，4为其数据坐在confidence_scores的每一维度数据
        filter = tf.where(score_filter)

        output = self.__get_session().run([filter,confidences,boxes], feed_dict={self._image_input_tensor: self._image_inputs})
        output_filter = output[0]
        output_confidences = output[1]
        output_bboxes = output[2]

        # 根据过滤器来获得每个属性（类，bbox，score）过滤后的tensor
        filtered_classes = []
        filtered_boxes = []
        filtered_cell= []
        filtered_score = []

        # 如果过滤器为空，则说明什么都没有检测出来
        if not output_filter.any():
            print('没有检测出任何物体')
            return None

        # 通过过滤器来获取类别，bbox，cell，score
        for i in range(len(output_filter)):
            temp = output_filter[i]
            filtered_classes.append(output_filter[i][3])
            filtered_boxes.append(output_bboxes[temp[0]][temp[1]][temp[2]][:])
            filtered_cell.append([temp[0],temp[1]])
            filtered_score.append(output_confidences[temp[0]][temp[1]][temp[2]])

        output_classes = []
        output_boxes = []
        output_cell = []
        output_scores = []

        # 数据分类
        for i in range(20):
            index = np.argwhere(np.array(filtered_classes)==i).reshape([-1])
            if index.size==0:
                continue
            # 每个类的类别
            temp = np.array(filtered_classes)[index].reshape([-1])
            output_classes.append(temp)
            # 每个类中每个检测对象的bbox
            temp = np.array(filtered_boxes)[index].reshape([-1,4])
            output_boxes.append(temp)
            # 每个类中每个检测对象的bbox所在的cell
            temp = np.array(filtered_cell)[index].reshape([-1, 2])
            output_cell.append(temp)
            # 每个类中的每个检测对象的得分
            temp = np.array(filtered_score)[index].reshape([-1])
            output_scores.append(temp)

        result_classes = []
        result_bboxes = []
        result_scores = []

        for i in range(len(output_classes)):
            classes, bboxes, scores=self.__interpert_result(output_classes[i],output_boxes[i],output_cell[i],output_scores[i])
            result_classes.append(classes)
            result_bboxes.append(bboxes)
            result_scores.append(scores)

        return result_classes,result_bboxes,result_scores

    def __interpert_result(self,classes,bboxes,cell,scores):
        '''
        解析输出结果的每个类的阈值
        :param classes:
        :param bboxes:
        :param cell:
        :param scores:
        :return:
        '''
        classes = np.array(classes)
        bboxes = np.array(bboxes)
        celles = np.array(cell)
        scores = np.array(scores)

        # 根据得分进行排序
        index = np.argsort(scores)[::-1] # 使用[::-1]来完成将序排列，或者np.argsort(-scores)
        classes = classes[index]
        bboxes = bboxes[index]
        celles = celles[index]
        scores = scores[index]

        # 将每个bbox转为绝对坐标
        for i in range(bboxes.shape[0]):
            bboxes[i] = self.__bbox_pos(celles[i], bboxes[i])
        # 通过非极大抑制，来筛选iou
        for i in range(len(classes)):
            if i==len(classes)-1:
                break
            else:
                for j in range(i+1,len(classes)):
                    iou = self.__iou(bboxes[i],bboxes[j])
                    if iou>=self._iou_threshold:
                        scores[j]=0

        return classes,bboxes,scores

    def __bbox_pos(self,cell,bbox):
        '''
        将相对坐标转化为绝对坐标
        :param cell:
        :param bbox:
        :return: bbox[left_top_x,left_top_y,right_bottom_x,right_bottom_y]
        '''

        # 计算bbox的1/2的宽度和高度
        half_bw = self._image_width*bbox[2]*bbox[2]/2
        half_bh = self._image_height*bbox[3]*bbox[3]/2

        # 计算原始大小的图片每个cell的宽和高
        cell_width = self._image_width/self._cell_size
        cell_height = self._image_height/self._cell_size

        # 计算中心坐标点
        center_x = bbox[0]*cell_width + cell_width*cell[1]
        center_y = bbox[1]*cell_height + cell_height*cell[0]

        return [center_x-half_bw,center_y-half_bh,center_x+half_bw,center_y+half_bh]

    def __iou(self,bbox1,bbox2):
        '''
        计算iou
        :param bbox1: [left_top_x,left_top_y,right_bottom_x,right_bottom_y]
        :param bbox2:[left_top_x,left_top_y,right_bottom_x,right_bottom_y]
        :return:
        '''
        bbox1_x = [bbox1[2], bbox1[0]]
        bbox1_y = [bbox1[3], bbox1[1]]
        bbox2_x = [bbox2[2], bbox2[0]]
        bbox2_y = [bbox2[3], bbox2[1]]

        # 根据4种判断条件来确定确定是否相交
        if bbox1_x[0] <= bbox2_x[1] or bbox1_y[0] <= bbox2_y[1] or bbox1_x[1] >= bbox2_x[0] or bbox1_y[1] >= bbox2_y[0]:
            return 0
        # 找到相交矩阵的坐标
        X = [max(bbox1_x[1], bbox2_x[1]), min(bbox1_x[0], bbox2_x[0])]
        Y = [max(bbox1_y[1], bbox2_y[1]), min(bbox1_y[0], bbox2_y[0])]

        # 计算交叉面积
        cross_area = float(abs(X[0]-X[1])*abs(Y[0]-Y[1]))
        # 计算bbox面积
        bbox1_area = abs(bbox1_x[0]-bbox1_x[1])*abs(bbox1_y[0]-bbox1_y[1])
        bbox2_area = abs(bbox2_x[0]-bbox2_x[1])*abs(bbox2_y[0]-bbox2_y[1])

        return cross_area / (bbox1_area + bbox2_area - cross_area)

    def __show_result(self,classes,bboxes,scores):
        '''
        显示结果
        :return:
        '''
        image = self._image.copy()
        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):
                if scores[i][j]:
                    cv2.rectangle(image, (bboxes[i][j][0], bboxes[i][j][1]), (bboxes[i][j][2], bboxes[i][j][3]), (0, 255, 0), 2)
                    cv2.rectangle(image, (int(bboxes[i][j][0]), int(bboxes[i][j][1] - 20)), (int(bboxes[i][j][2]), int(bboxes[i][j][1])), (125, 125, 125), -1)
                    cv2.putText(image, self.classes[classes[i][j]] + ' : %.2f' % scores[i][j], (int(bboxes[i][j][0]+5), int(bboxes[i][j][1]-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow(self._net_name+'detection', image)
        cv2.waitKey(3000)

    def train(self,loss):
        '''
        网络训练
        :param loss:
        :return:
        '''
        train_option = tf.train.MomentumOptimizer(self.__learning_rate,self.__momentum).minimize(loss)
        return train_option

    def loss(self):
        '''
        计算网络的loss
        :return:
        '''
        # 获取每个类的概率值,置信度,bbox
        classes_probs = tf.reshape(self._net_output[:, 0:7 * 7 * 20], [-1,7, 7, 20])
        confidences = tf.reshape(self._net_output[:, 7 * 7 * 20:7 * 7 * 22], [-1,7, 7, 2])
        boxes = tf.reshape(self._net_output[:, 7 * 7 * 22:], [-1,7, 7, 2, 4])




