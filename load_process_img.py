# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:34:08 2019

@author: Shilpa
"""

import imageio
from skimage.transform import resize
import numpy as np

"""
Load and resize the image
Inputs: 
- filename: Path to the image file.
- size: The new size of the image. Here the new image has same height and width

Returns:
A matrix of shape [size,size,3] with values from range 0 to 255  
"""
def load_image(filename,size=None) :
    img = imageio.imread(filename)
    if size is not None :
        img = resize(img,[size,size])
    return img * 255.0

VGG_MEAN = np.array([123.68,116.779,103.939], dtype=np.float32)

"""
Subtracts image pixel values from VGG mean
Inputs: 
-img : A matrix of shape [size,size,3]

Returns:
A matrix of same shape as input with RGB values subtraced by VGG19 mean values  
"""
def preprocess_image(img) :
    return img.astype(np.float32) - VGG_MEAN
    
"""
Add the VGG 19 mean to RGB values and clip the values to range [0-255]
Inputs: 
-img : A matrix of shape [size,size,3]

Returns:
A matrix of same shape as input with RGB values added with VGG19 mean values and clipped to fit the range of 0-255  
"""
def deprocess_image(img) :
    img = img + VGG_MEAN
    return np.clip(img,0.0,255.0).astype(np.uint8)