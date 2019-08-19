from pycocotools import coco
import numpy as np
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import pylab
import random
import cv2
import aauRainSnowUtility

if __name__ == '__main__':
    rgbAnnFile = 'aauRainSnow-rgb.json'
    thermalAnnFile = 'aauRainSnow-thermal.json'

    rainSnowRgbGt = coco.COCO(rgbAnnFile)
    rainSnowThermalGt = coco.COCO(thermalAnnFile)

    # Plot an image
    chosenImgId = random.randint(0, 2167)
    annIds = rainSnowRgbGt.getAnnIds(imgIds=[chosenImgId])
    anns = rainSnowRgbGt.loadAnns(annIds)
    thermalImg = rainSnowThermalGt.loadImgs([chosenImgId])[0]
    rgbImg = rainSnowRgbGt.loadImgs([chosenImgId])[0]

    I = io.imread(thermalImg['file_name'])
    # Transform thermal img to match rgb
    I = aauRainSnowUtility.transferCamera(I, thermalImg['file_name'])
    plt.gcf().clear()
    plt.axis('off')
    plt.imshow(I)
    rainSnowRgbGt.showAnns(anns)
    plt.show()

    I = io.imread(rgbImg['file_name'])
    plt.gcf().clear()
    plt.axis('off')
    plt.imshow(I)
    rainSnowRgbGt.showAnns(anns)
    plt.show()
