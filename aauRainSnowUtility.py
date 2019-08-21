import os

import cv2


# For additional methods refer to https://www.kaggle.com/aalborguniversity/aau-rainsnow

def getRegistrationVarsFromFileName(imageFileName: str):
    split = imageFileName.split('/')

    if len(split) >= 3:
        scene = split[0]
        sequence = split[1]
        base_folder = 'dataset/val' if scene == 'Hadsundvej' else 'dataset/train'

        # Then load the registration variables manually
        print("Reading calibration file at: " + os.path.join(scene, sequence + '-calib.yml'))
        fs = cv2.FileStorage(os.path.join(base_folder, os.path.join(scene, sequence + '-calib.yml')), cv2.FILE_STORAGE_READ)

        registration = dict()
        registration["homCam1Cam2"] = fs.getNode("homCam1Cam2").mat()

        registration["homCam2Cam1"] = fs.getNode("homCam2Cam1").mat()
        registration["cam1CamMat"] = fs.getNode("cam1CamMat").mat()
        registration["cam2CamMat"] = fs.getNode("cam2CamMat").mat()

        registration["cam1DistCoeff"] = fs.getNode("cam1DistCoeff").mat()
        registration["cam2DistCoeff"] = fs.getNode("cam2DistCoeff").mat()

        return registration
    else:
        return None


def transferCamera(img, img_name, mode='thermalToRgb'):
    '''
    :param img: image to transfer, as matrix
    :param img_name: image name as string
    :param mode: can either be thermalToRgb or rgbToThermal
    :return: image that is transferred
    '''
    registration = getRegistrationVarsFromFileName(img_name)
    homography = registration["homCam1Cam2"] if mode == 'rgbToThermal' else registration["homCam2Cam1"]

    transformed_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

    return transformed_img
