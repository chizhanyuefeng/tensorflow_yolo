import numpy as np
#from src.yolo_net.net import Net
from net import Net
import tensorflow as tf

class YoloTinyNet(Net):

    def __init__(self, trainable = False):
        super(YoloTinyNet,self).__init__()
        self._cfg_file_path = '../../cfg/yolo_tiny.cfg'
        self._model_path = '../../weights/yolo_tiny/model.ckpt'#'../../weights/YOLO_tiny.ckpt'
        self._net_name = 'Yolo tiny net'
        self._trainable = trainable
        self._construct_graph()

if __name__ =='__main__':
    # tiny = YoloTinyNet(trainable = False)
    # tiny.test('../../data/dog.jpg')
    # tiny = YoloTinyNet(trainable = True)
    # tiny.train()

    a = tf.constant(0,tf.float32)
    sess = tf.Session()
    b = 1/a


    #sess.run(tf.global_variables_initializer())

    print(sess.run(b))