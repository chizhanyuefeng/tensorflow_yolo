import numpy as np
from src.yolo_net.net import Net
import tensorflow as tf

class YoloTinyNet(Net):

    def __init__(self):
        super(YoloTinyNet,self).__init__()
        self._cfg_file_path = '../../cfg/tiny-yolo.cfg'
        self._model_path = '../../weights/YOLO_tiny.ckpt'

    def train(self):
        # TODO:
        pass

    def loss(self):
        # TODO:
        pass

if __name__ =='__main__':
    tiny = YoloTinyNet()
    tiny.construct_graph()
    tiny.load_model()
    tiny.test('../../data/123.jpg')


    # a = np.zeros([3,3])
    # a[:,0:1] = 1
    # print(a)
    # b = np.nonzero(a)
    # print(b)

    # a = tf.Variable([[0,0],[1,1]])
    # c = tf.constant([[1],[2]])
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(a))
    #
    # b = a[:,0:1]
    # b = tf.multiply(b,c)
    # print(sess.run(a))













