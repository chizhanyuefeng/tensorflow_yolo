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


if __name__ =='__main__':
    # tiny = YoloTinyNet()
    # tiny.test('../../data/car.jpg')
    a = tf.constant([[0,2],[3,0]])
    b = a+1
    sess = tf.Session()
    print(sess.run(b))















