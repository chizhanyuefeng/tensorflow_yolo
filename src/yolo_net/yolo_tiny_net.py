import numpy as np
from src.yolo_net.net import Net
import tensorflow as tf

class YoloTinyNet(Net):

    def __init__(self):
        super(YoloTinyNet,self).__init__()
        self._cfg_file_path = '../../cfg/tiny-yolo.cfg'
        self._model_path = '../../weights/YOLO_tiny.ckpt'
        self._net_name = 'Yolo tiny net'
        self._construct_graph()

    def train(self):
        # TODO:
        pass

    def loss(self):
        # TODO:
        pass

if __name__ =='__main__':
    tiny = YoloTinyNet()
    #tiny.construct_graph()
    #tiny.load_model()
    tiny.test('../../data/cute.jpeg')


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

    # a = np.array([2,3,1])
    # b = np.argsort(a)
    # print(b)
    #
    # a = tf.constant([False,False])
    # b = tf.where(a)
    # sess = tf.Session()
    # c = sess.run(b)
    # if not c:
    #     print('dwadwa')
    # else:
    #     print('dwadawdawdwadwad')

















