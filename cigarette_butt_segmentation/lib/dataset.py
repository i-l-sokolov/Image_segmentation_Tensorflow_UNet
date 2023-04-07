import os
from utils import get_mask
from glob import glob
import json
from PIL import Image
import numpy as np
import tensorflow as tf

def get_dataset(folder):
    """

    Parameters
    ----------
    folder str, 'train' or 'val'

    Returns
    -------
    tensorflow dataset
    """
    annotations = json.load(open(f'data/{folder}/coco_annotations.json','r'))
    images_names = sorted(glob(f'data/{folder}/images/*'))[:100]
    if folder == 'val':
        images = [np.array(Image.open(image).convert('RGB')) / 255 for image in images_names]
    elif folder == 'train':
        images = [np.array(Image.open(image)) / 255 for image in images_names]
    images_ids =  [int(x.split('/')[-1].split('.')[0]) for x in images_names]
    masks = [get_mask(image_id, annotations).reshape(512,512,1) / 255 for image_id in images_ids]
    return tf.data.Dataset.from_tensor_slices((images,masks))
