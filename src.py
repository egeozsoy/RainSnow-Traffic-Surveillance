import tensorflow as tf
from pycocotools import coco
from preprocess import load_and_preprocess_image,load_tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

# TODO tensorflow Dataset class for loading data(a lot of work is already done in preprocess)
# TODO decide on a that fusion that makes sense aka. how to represent data(features and labels)
# TODO find meaningful augmentations that we can apply
# TODO decide on an initial model architecture(unet, mask-rcnn)
#https://www.tensorflow.org/beta/guide/effective_tf2
#https://www.tensorflow.org/guide/eager

#https://www.tensorflow.org/beta/tutorials/load_data/images

if __name__ == '__main__':
    rgbAnnFile = 'aauRainSnow-rgb.json'
    thermalAnnFile = 'aauRainSnow-thermal.json'

    rainSnowRgbGt = coco.COCO(rgbAnnFile)
    rainSnowThermalGt = coco.COCO(thermalAnnFile)

    image_indices = [img['id'] for img in rainSnowThermalGt.imgs.values()]
    path_ds = tf.data.Dataset.from_tensor_slices((image_indices))
    image_ds = path_ds.map(load_tf, num_parallel_calls=1)

    print(next(iter(image_ds)))

    print('end')