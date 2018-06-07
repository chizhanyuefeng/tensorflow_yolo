import numpy as np
from src.net import Net
import tensorflow as tf

class YoloTinyNet(Net):
    def __init__(self, trainable=False, download_model=False):
        super(YoloTinyNet,self).__init__()
        self._cfg_file_path = './cfg/yolo_tiny.cfg'

        if download_model:
            self._model_path = './weights/YOLO_tiny.ckpt'
            self.WIGHTS_NAME = False
            print('????????????????????????????????????????????????????????')
        else:
            self._model_path = './weights/yolo_tiny/model.ckpt'
            self.WIGHTS_NAME = True

        self._net_name = 'Yolo tiny net'
        self._trainable = trainable
        self._inference()

if __name__ =='__main__':
    # tiny = YoloTinyNet(trainable=False)
    # tiny.test('../data/dog.jpg')
    tiny = YoloTinyNet(trainable=True)
    tiny.train()

    # a = tf.constant(64,tf.float32,name='a')
    # sess = tf.Session()
    # b = 70%a/a
    # print(sess.run(b))