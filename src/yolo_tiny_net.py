import numpy as np
from src.net import Net
import tensorflow as tf

class YoloTinyNet(Net):
    def __init__(self, trainable=False):
        super(YoloTinyNet,self).__init__()
        self._cfg_file_path = './cfg/yolo_tiny.cfg'
        self._model_path = './weights/yolo_tiny/model.ckpt'#'../weights/YOLO_tiny.ckpt'
        self._net_name = 'Yolo tiny net'
        self._trainable = trainable
        self._construct_graph()

if __name__ =='__main__':
    # tiny = YoloTinyNet(trainable=False)
    # tiny.test('../data/dog.jpg')
    tiny = YoloTinyNet(trainable=True)
    tiny.train()

    # a = tf.constant(64,tf.float32,name='a')
    # sess = tf.Session()
    # b = 70%a/a
    # print(sess.run(b))