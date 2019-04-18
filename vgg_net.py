# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:30:58 2019

@author: Shilpa
"""
import h5py
import tensorflow as tf

"""
Extract VGG features from an image
Inputs: 
- img: A matrix of shape [size,size,3]
- h5_file: Path to VGG weights file.

Returns:
An array having the activation values of all the VGG 19 layers excpet the top fully connected layers
"""
def extract_features(img,h5_file) :
    file = h5py.File(h5_file,'r')
    layers = []
    x = tf.convert_to_tensor(img,dtype=tf.float32)
    with tf.variable_scope('vgg19',reuse=tf.AUTO_REUSE) :
        with tf.variable_scope('block1') :
            with tf.variable_scope('conv1') :
                group = file['block1_conv1']
                W = tf.constant(group['block1_conv1_W_1:0'])
                b = tf.constant(group['block1_conv1_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv2') :
                group = file['block1_conv2']
                W = tf.constant(group['block1_conv2_W_1:0'])
                b = tf.constant(group['block1_conv2_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('block1_pool') :
                x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
                layers.append(x)
        with tf.variable_scope('block2') :
            with tf.variable_scope('conv1') :
                group = file['block2_conv1']
                W = tf.constant(group['block2_conv1_W_1:0'])
                b = tf.constant(group['block2_conv1_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv2') :
                group = file['block2_conv2']
                W = tf.constant(group['block2_conv2_W_1:0'])
                b = tf.constant(group['block2_conv2_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('block2_pool') :
                x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
                layers.append(x)
        with tf.variable_scope('block3') :
            with tf.variable_scope('conv1') :
                group = file['block3_conv1']
                W = tf.constant(group['block3_conv1_W_1:0'])
                b = tf.constant(group['block3_conv1_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv2') :
                group = file['block3_conv2']
                W = tf.constant(group['block3_conv2_W_1:0'])
                b = tf.constant(group['block3_conv2_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv3') :
                group = file['block3_conv3']
                W = tf.constant(group['block3_conv3_W_1:0'])
                b = tf.constant(group['block3_conv3_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv4') :
                group = file['block3_conv4']
                W = tf.constant(group['block3_conv4_W_1:0'])
                b = tf.constant(group['block3_conv4_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('block3_pool') :
                x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
                layers.append(x)
        with tf.variable_scope('block4') :
            with tf.variable_scope('conv1') :
                group = file['block4_conv1']
                W = tf.constant(group['block4_conv1_W_1:0'])
                b = tf.constant(group['block4_conv1_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv2') :
                group = file['block4_conv2']
                W = tf.constant(group['block4_conv2_W_1:0'])
                b = tf.constant(group['block4_conv2_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv3') :
                group = file['block4_conv3']
                W = tf.constant(group['block4_conv3_W_1:0'])
                b = tf.constant(group['block4_conv3_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv4') :
                group = file['block4_conv4']
                W = tf.constant(group['block4_conv4_W_1:0'])
                b = tf.constant(group['block4_conv4_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('block4_pool') :
                x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
                layers.append(x)
        with tf.variable_scope('block5') :
            with tf.variable_scope('conv1') :
                group = file['block5_conv1']
                W = tf.constant(group['block5_conv1_W_1:0'])
                b = tf.constant(group['block5_conv1_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv2') :
                group = file['block5_conv2']
                W = tf.constant(group['block5_conv2_W_1:0'])
                b = tf.constant(group['block5_conv2_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv3') :
                group = file['block5_conv3']
                W = tf.constant(group['block5_conv3_W_1:0'])
                b = tf.constant(group['block5_conv3_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+ b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('conv4') :
                group = file['block5_conv4']
                W = tf.constant(group['block5_conv4_W_1:0'])
                b = tf.constant(group['block5_conv4_b_1:0'])
                x = tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b
                x = tf.nn.relu(x)
                layers.append(x)
            with tf.variable_scope('block5_pool') :
                x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
                layers.append(x)
        return layers