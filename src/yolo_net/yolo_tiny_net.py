import numpy as np
from src.yolo_net.net import Net


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
    # tiny = YoloTinyNet()
    # tiny.construct_graph()
    # tiny.load_model()
    # tiny.test('../../data/dog.jpg')


    a = np.zeros([3,3,5])
    a[:, :,2:3] = 1
    b = np.ones([3,3,2])

    c = np.multiply(a,b)
    print(c.shape)









