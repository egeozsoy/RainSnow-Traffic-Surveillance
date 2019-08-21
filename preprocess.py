import numpy as np
import skimage.io as io
import tensorflow as tf
from pycocotools import coco

import aauRainSnowUtility

rgbAnnFile = 'aauRainSnow-rgb.json'
thermalAnnFile = 'aauRainSnow-thermal.json'

rainSnowRgbGt = coco.COCO(rgbAnnFile)
rainSnowThermalGt = coco.COCO(thermalAnnFile)


def load_and_preprocess_image(image_id):
    if not isinstance(image_id, int):
        image_id = image_id.numpy()
    annIds = rainSnowRgbGt.getAnnIds(imgIds=[image_id])
    anns = rainSnowRgbGt.loadAnns(annIds)
    thermalImg_info = rainSnowThermalGt.loadImgs([image_id])[0]
    rgbImg_info = rainSnowRgbGt.loadImgs([image_id])[0]

    thermal_image = io.imread('Dataset/train/' + thermalImg_info['file_name'])
    # Transform thermal img to match rgb
    thermal_image = aauRainSnowUtility.transferCamera(thermal_image, thermalImg_info['file_name'])[:, :, :1]  # all channels are same, ignore the others

    rgb_image = io.imread(rgbImg_info['file_name'])
    fused_image = np.concatenate([rgb_image, thermal_image], axis=2)

    categories = []
    binary_anns = []
    for ann in anns:
        if ann['segmentation'] != []:
            categories.append(ann['category_id'])
            binary_anns.append(rainSnowRgbGt.annToMask(ann))

    return fused_image, tf.convert_to_tensor(categories), tf.convert_to_tensor(binary_anns)


def load_tf(image_id):
    return tf.py_function(load_and_preprocess_image, [image_id], [tf.float32, tf.int32, tf.int32])


if __name__ == '__main__':
    # Example loading and process and overlaying images on top of each other
    load_and_preprocess_image(0)
