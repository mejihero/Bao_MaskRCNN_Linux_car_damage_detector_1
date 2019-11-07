#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:06:23 2019

@author: luy1
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
#ROOT_DIR = os.path.abspath("../")
ROOT_DIR = '/home/luy1/Desktop/Mask_RCNN-master'



# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#%matplotlib inline 

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = '/home/luy1/Desktop/experiment-2/logs_2'

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = '/home/luy1/Desktop/experiment-2/logs_2/part20190401T1359/mask_rcnn_part_0002.h5'


# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = '/home/luy1/Desktop/experiment-2/dataset/test'

sys.path.append('/home/luy1/Desktop/experiment-2')
import custom_multiclass_5


config = custom_multiclass_5.BalloonConfig()
#custom_DIR = os.path.join(ROOT_DIR, "customImages")


class InferenceConfig(config.__class__):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ["rear_bumper", "front_bumper", "headlamp", "door", "hood"]


file_names = next(os.walk(IMAGE_DIR))[2]
file_names


file_name = random.choice(file_names)
print(file_name)

image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
image


results = model.detect([image], verbose =1)

r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                           class_names, file_name, r['scores'])



#loop the prediction
for name in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, name))
    results = model.detect([image], verbose = 1)
    r = results[0]
    visualize.display_instances(image,
                               r['rois'], r['masks'], r['class_ids'],
                               class_names, name, r['scores'])



'/home/luy1/Desktop/experiment-2/dataset/infer/'+file_name+'.jpg'


###############################################################################################
    
import sys
print(sys.path)    
    
    
import mrcnn
print(mrcnn.__file__)


print(visualize.display_instances().__file__)
print(visualize.display_instances())

import sys
sys.path.append('/home/luy1/Desktop/experiment-2')

import predict





#########################################################################################
import cv2



def display_results(image, boxes, masks, class_ids, class_names, scores=None,
                        show_mask=True, show_bbox=True, display_img=True,
                        save_img=True, save_dir=None, img_name=None):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset (Without Background)
        scores: (optional) confidence scores for each box
        show_mask, show_bbox: To show masks and bounding boxes or not
        display_img: To display the image in popup
        save_img: To save the predict image
        save_dir: If save_img is True, the directory where you want to save the predict image
        img_name: If save_img is True, the name of the predict image

        """
        n_instances = boxes.shape[0]
        colors = color_map()
        for k in range(n_instances):
            color = colors[class_ids[k]].astype(np.int)
            if show_bbox:
                box = boxes[k]
                cls = class_names[class_ids[k]-1]  # Skip the Background
                score = scores[k]
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), color, 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, '{}: {:.3f}'.format(cls, score), (box[1], box[0]),
                            font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

            if show_mask:
                mask = masks[:, :, k]
                color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.int)
                color_mask[mask] = color
                image = cv2.addWeighted(color_mask, 0.5, image.astype(np.int), 1, 0)

        if display_img:
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
        if save_img:
            cv2.imwrite(os.path.join(save_dir, img_name), image)

        return None



def color_map(N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap


display_results(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], 
                save_dir="/home/luy1/Desktop/experiment-2/dataset/infer", 
                img_name="inferenced.png")




import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
#plt.show()
plt.savefig('/home/luy1/Desktop/experiment-2/dataset/infer/demo.jpg')























