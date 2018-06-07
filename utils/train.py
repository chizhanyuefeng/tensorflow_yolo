'''
训练图片前，产生的label
'''

import cv2
import numpy as np

def load_img(path):
    img = cv2.imread(path)
    img_height, img_width, _ = img.shape

    return img_height, img_width

def make_label(img_height, img_width, labels):
    yolo_label = []
    height_scale = 448 / img_height
    width_scale = 448 / img_width

    for label in labels:
        height = (label[3] - label[1]) * height_scale
        width = (label[2] - label[0]) * width_scale
        x = (label[0] + width / 2) * width_scale
        y = (label[1] + height / 2) * height_scale
        yolo_label.append([x, y, width, height, label[4]])

    return yolo_label

label = [[128,222,325,541,11],[463,62,686,172,6],[118,132,567,425,1]]

img_height, img_width = load_img('../data/dog.jpg')
yolo_label = make_label(img_height, img_width, label)

print(yolo_label)