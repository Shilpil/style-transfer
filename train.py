# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:35:41 2019

@author: Shilpa
"""
import tensorflow as tf
import imageio
import vgg_net as vgg
import loss
import load_process_img as pre_img

"""
Generates the stylized image
Inputs: 
-options: Options provided by the user

Returns:
Has no return value

Calculates the total loss and uses Adam optimizer 
Prints the 3 losses onto the console.
Optionally prints the intermediate images generated.
Finally prints the stylized image into the output folder 
"""
def train(options) :
    content_weight = options.content_weight
    tv_weight = options.tv_weight
    initial_lr = options.initial_lr
    max_iter = options.max_iter
    style_weights = options.style_weights
    print_iterations = options.print_iterations
    img_size = options.img_size
    content = options.content
    style = options.style
    output = options.output
    beta1 = options.beta1
    beta2 = options.beta2
    epsilon = options.epsilon
    h5_file = options.h5_file
    
    content_layer = 12
    style_layers = [0, 3, 6, 11, 16]
    style_target_vars = []
    print(tf.test.is_gpu_available(cuda_only=True))
    contentImg  = pre_img.load_image(content,size=img_size) 
    contentImg = pre_img.preprocess_image(contentImg)
    styleImg  = pre_img.load_image(style,size=img_size)
    styleImg = pre_img.preprocess_image(styleImg)
    img_var = tf.Variable(contentImg[None], name="image",dtype=tf.float32)
    lr_var = tf.Variable(initial_lr, name="lr")
    new_img_feats = vgg.extract_features(img_var,h5_file)
    content_img_feats = vgg.extract_features(contentImg[None],h5_file)
    style_img_feats = vgg.extract_features(styleImg[None],h5_file)
    for idx in style_layers :
        style_target_vars.append(loss.gram_matrix(style_img_feats[idx]))
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        style_loss = loss.style_loss(new_img_feats,style_layers,style_target_vars,style_weights)
        content_loss = loss.content_loss(content_weight,new_img_feats[content_layer],content_img_feats[content_layer])
        tv_loss = loss.tv_loss(img_var,tv_weight)
        total_loss = style_loss + content_loss + tv_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_var,beta1=beta1,beta2=beta2,epsilon=epsilon)
        training_op = optimizer.minimize(total_loss,var_list=[img_var])
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(img_var.initializer)
        for t in range(max_iter) :
            if print_iterations is not None and t % print_iterations == 0 :
                new_image = img_var.eval()
                imageio.imwrite(output + '\\iteration_' + str(t) + '.jpg',pre_img.deprocess_image(new_image[0]))
            sess.run(training_op)
            loss_val = sess.run(total_loss)
            s_loss = sess.run(style_loss)
            c_loss = sess.run(content_loss)
            print(str(t) + ':' + str(loss_val) + '\t' + str(s_loss) + '\t' + str(c_loss))
        new_image = sess.run(img_var)
        imageio.imwrite(output + '\\final.jpg',pre_img.deprocess_image(new_image[0]))