import numpy as np
from src.yolo_net.net import Net
import tensorflow as tf

class YoloTinyNet(Net):

    def __init__(self, trainable = False):
        super(YoloTinyNet,self).__init__()
        self._cfg_file_path = '../../cfg/yolo_tiny.cfg'
        self._model_path = '../../weights/YOLO_tiny.ckpt'
        self._net_name = 'Yolo tiny net'
        self._trainable = trainable
        self._construct_graph()


if __name__ =='__main__':
    tiny = YoloTinyNet(True)
    #tiny.test('../../data/2.jpg')
    tiny.train()

    # a = tf.zeros([3,3,2])
    # sess = tf.Session()
    # a[2, 2, :] = [1, 2]
    # print(sess.run(a))
    # a = tf.ones([1,2,3,4],tf.float32)
    # b = tf.constant([1,0,3,4],tf.float32)
    # c = 2-a
    # #b = tf.transpose(a,[1,2,3,0])
    #
    # sess = tf.Session()

    #print(sess.run(c))