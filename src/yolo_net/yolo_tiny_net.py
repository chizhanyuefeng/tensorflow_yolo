import net
import configparser as cp
class YoloTinyNet(net.Net):

    def __init__(self,net_cfg_file):
        cfg = cp.ConfigParser(net_cfg_file)
        print(cfg)


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
    tiny = YoloTinyNet('../cfg/tiny-yolo.cfg')



