import tensorflow as tf
from pycocotools import coco
from preprocess import load_and_preprocess_image, load_tf
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import skimage
from matplotlib import pyplot as plt

import zipfile
import urllib.request
import shutil
import os

import numpy as np

# TODO find meaningful augmentations that we can apply

# TODO tutorial and sample code with MASK_RCNN
# TODO check this for dataformat etc. https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

# https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/coco.py (might be helpful)

# Architecture to use is ttps://github.com/matterport/Mask_RCNN

############################################################
#  Configurations
############################################################


class_name_mapping = {0: 'empty', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'plane', 6: 'bus', 7: 'train', 8: 'truck'}


class RainSnowTrafficConfig(Config):
    """Configuration for training on RainSnowTraffic.
    Derives from the base Config class and overrides values
    """
    # Give the configuration a recognizable name
    NAME = "RainSnowTraffic"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # RainSnowTraffic has 6 classes


############################################################
#  Dataset
############################################################

rainSnowRgbGt = coco.COCO('aauRainSnow-rgb.json')


class RainSnowTrafficDataset(utils.Dataset):
    def load_rainsnowtraffic(self, dataset_dir, subset):
        """Load the dataset
        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val, minival, valminusminival) # TODO Not implemented yet
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        # TODO merge thermal images
        coco = COCO('aauRainSnow-rgb.json')
        image_dir = "{}/{}".format(dataset_dir, subset)
        # All classes
        class_ids = sorted(coco.getCatIds())[:8]

        # All images
        image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            if i == 5 or i == 7: # we don't have any of these classes, but not sure if this is a correct fix
                continue
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            path = os.path.join(image_dir, coco.imgs[i]['file_name'])
            if os.path.exists(path):
                self.add_image(
                    "coco", image_id=i,
                    path=path,
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annotations=coco.loadAnns(coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id and annotation['segmentation'] != []:
                m = rainSnowRgbGt.annToMask(annotation)

                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(RainSnowTrafficDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return info['path']
        else:
            super(RainSnowTrafficDataset, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = RainSnowTrafficDataset()
    dataset_train.load_rainsnowtraffic('dataset', 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RainSnowTrafficDataset()
    dataset_val.load_rainsnowtraffic('dataset', 'val')
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def display_instances(image, masks, class_ids):
    """
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = masks.shape[2]

    # If no axis is passed, create one and automatically call show()
    _, ax = plt.subplots(1, figsize=(16, 16))
    # Generate random colors
    colors = visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title("")

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        class_id = class_ids[i]
        label = class_name_mapping[class_id]
        caption = label

        # Mask
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.show()


def detect_and_visualize(model, image_path):
    image = skimage.io.imread(image_path)
    r = model.detect([image], verbose=1)[0]
    display_instances(image, r['masks'], r['class_ids'])


if __name__ == '__main__':
    TRAINING: bool = True
    image_path = ''
    LOGS_DIR = "logs"

    # Configurations
    if TRAINING:
        config = RainSnowTrafficConfig()
    else:
        class InferenceConfig(RainSnowTrafficConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()

    config.display()

    # Create model
    if TRAINING:
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=LOGS_DIR)

    # Load weights
    print("Loading weights")
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if TRAINING:
        train(model)
    else:
        detect_and_visualize(model, 'dataset/val/Hadsundvej/Hadsundvej-1/cam1-00239.png')
