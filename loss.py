# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:32:49 2019

@author: Shilpa
"""
import tensorflow as tf

"""
Compute the content loss for style transfer.

Inputs:
- content_weight: scalar constant we multiply the content_loss by.
- content_current: features of the current image, Tensor with shape [1, height, width, channels]
- content_target: features of the content image, Tensor with shape [1, height, width, channels]

Returns:
- scalar content loss
"""
def content_loss(content_weight,content_current,content_original) :
    return content_weight * tf.reduce_sum(tf.squared_difference(content_current,content_original))

"""
Compute the Gram matrix from features.

Inputs:
- features: Tensor of shape (1, H, W, C) giving features for
  a single image.
- normalize: optional, whether to normalize the Gram matrix
    If True, divide the Gram matrix by the number of neurons (H * W * C)

Returns:
- gram: Tensor of shape (C, C) giving the (optionally normalized)
  Gram matrices for the input image.
"""
def gram_matrix(features, normalize=True) :
    shape_m = tf.shape(features)
    reshaped_features = tf.reshape(features,shape=(shape_m[0]*shape_m[1]*shape_m[2],shape_m[3]))
    gram_m = tf.tensordot(tf.transpose(reshaped_features),reshaped_features,axes = 1)
    if(normalize) :
        gram_m = gram_m / tf.to_float(shape_m[0]*shape_m[1]*shape_m[2]*shape_m[3])
    return gram_m

"""
Computes the style loss at a set of layers.

Inputs:
- new_img_feats: list of the features at every layer of the current image, as produced by
  the extract_features function.
- style_layers: List of layer indices into feats giving the layers to include in the
  style loss.
- style_gram: List of the same length as style_layers, where style_targets[i] is
  a Tensor giving the Gram matrix of the source style image computed at
  layer style_layers[i].
- style_weights: List of the same length as style_layers, where style_weights[i]
  is a scalar giving the weight for the style loss at layer style_layers[i].
  
Returns:
- style_loss: A Tensor containing the scalar style loss.
"""
def style_loss(new_img_feats,style_layers,style_gram,style_weights) :
    style_loss_var = tf.Variable(0,'style_loss',dtype = tf.float32)
    for idx, val in enumerate(style_layers) :
        new_img_layer_gram = gram_matrix(new_img_feats[val])
        style_layer_gram = style_gram[idx]
        diff = tf.reduce_sum(tf.squared_difference(new_img_layer_gram,style_layer_gram))
        weighed_diff = style_weights[idx] * diff
        style_loss_var = style_loss_var + weighed_diff
    return style_loss_var

"""
Compute total variation loss.

Inputs:
- img: Tensor of shape (1, H, W, 3) holding an input image.
- tv_weight: Scalar giving the weight w_t to use for the TV loss.

Returns:
- loss: Tensor holding a scalar giving the total variation loss
  for img weighted by tv_weight.
"""
def tv_loss(img,tv_weight) :
    shape_m = tf.shape(img)
    img_h1 = tf.slice(img,[0,0,0,0],[shape_m[0],shape_m[1]-1,shape_m[2],shape_m[3]])
    img_h2 = tf.slice(img,[0,1,0,0],[shape_m[0],shape_m[1]-1,shape_m[2],shape_m[3]])
    img_v1 = tf.slice(img,[0,0,0,0],[shape_m[0],shape_m[1],shape_m[2]-1,shape_m[3]])
    img_v2 = tf.slice(img,[0,0,1,0],[shape_m[0],shape_m[1],shape_m[2]-1,shape_m[3]])
    loss = tv_weight * (tf.reduce_sum(tf.squared_difference(img_h1,img_h2)) + tf.reduce_sum(tf.squared_difference(img_v1,img_v2)))
    return loss