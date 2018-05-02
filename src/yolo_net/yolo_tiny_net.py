from src.yolo_net.net import Net

class YoloTinyNet(Net):

    def __init__(self,net_cfg_file):
        self._cfg_file_path = net_cfg_file

    def test(self):
        #TODO:
        pass

    def train(self):
        # TODO:
        pass

    def loss(self):
        # TODO:
        pass

if __name__ =='__main__':
    tiny = YoloTinyNet('../../cfg/tiny-yolo.cfg')
    tiny.construct_graph()
    tiny.load_model('../../weights/YOLO_tiny.ckpt')




