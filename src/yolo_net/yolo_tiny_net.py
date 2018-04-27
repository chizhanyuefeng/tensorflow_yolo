import net
from optparse import OptionParser

class YoloTinyNet(net.Net):

    def __init__(self,net_cfg_file):

        pass

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
    c = [{'a':1},{'b':2}]



