# style-transfer
Simple Neural Style transfer using pretrained VGG19 weights

This is a simple implementation of style transfer based on the paper : A Neural Algorithm of Artistic Style (https://arxiv.org/abs/1508.06576)
It is based on neural networks and uses VGG 19

The implementation is as follows:
1. Load and preprocess the content and style image
2. Initialise the final image with the content image
3. Extract the features of all the 3 images (Content, Style and new generated image)
4. Calculate the loss which is a sum of Content loss + Style loss + Total variation loss.
5. Use Adam optimizer to reduce the loss and modify the new generated image.

Usually the inputs are kept constant and the weights are varied. Here the weights are kept constant and the input(New image generated) is varied.
Here the weights of the VGG 19 network are kept constant to reduce the degrees of freedom
The weights is a H5 file downloaded from URL :
https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5

Loss calculation :
Content loss: Content loss measures how much the feature map of the generated image differs from the feature map of the source image
It is the weighted euclidiean distance between the pixel values of the two images. The activation values of the 13th layer (Block_4_Conv 2) is used.

Style loss : 
Calculated using the Gram matrix which represents the correlations between the responses of each filter. The Gram matrix is an approximation to the covariance matrix -- we want the activation statistics of our generated image to match the activation statistics of our style image, and matching the (approximate) covariance is one way to do that.
The style loss for the layer is simply the weighted Euclidean distance between the two Gram matrices.
The total style loss is the sum of style losses at each layer.
Layers used : Block_1_Conv_1 , Block_2_Conv_1 , Block_3_Conv_1, Block_4_Conv_1, Block_5_Conv_1

Total variation loss :
It the weighted sum of the squares of differences in the pixel values for all pairs of pixels that are next to each other (horizontally or vertically)

Here all the values of weights used for the losses are taken from the arguments given by the user(If not provided the default values are taken). Varying these values generates different images.In fact varying the weights for different layers of the style loss also generates different images.
