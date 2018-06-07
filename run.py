import tensorflow as tf
from src.yolo_tiny_net import *

flags = tf.flags

flags.DEFINE_bool('train', False, 'If need to train?True or False')
flags.DEFINE_string('yolo_model', 'tiny', 'There has three yolo models: tiny, small, normal')
flags.DEFINE_string('test_img', './data/dog.jpg', 'where the test image is stored.')
flags.DEFINE_bool('debug', False, 'If need to debug?True or False')
FLAGS = flags.FLAGS


def main(_):
    if FLAGS.train:
        yolo = YoloTinyNet(True)
        yolo.train()
    else:
        yolo = YoloTinyNet(False)
        yolo.test(FLAGS.test_img)

if __name__=='__main__':
    tf.app.run()