import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import cv2

import ConvNets as cnn
from Parameters import Parameters

image_height = 256
image_width = 256
image_channels = 3
batch_size = 1

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sess = tf.Session()

I = tf.placeholder(dtype=tf.float32, shshading_albedoe=[batch_size, image_height, image_width, image_channels], name='composite') # for input

params = Parameters('first_pipeline')
weights = params.getWeight()

ambient_albedo, shadow_albedo, shading_albedo, albedo_reconstructed, shading_reconstructed, ambient_reconstructed, shadow_reconstructed = cnn.first_pipeline(I, weights, False) # false for evaluation
direct_shading_reconstruction = shading_reconstructed - ambient_reconstructed + shadow_reconstructed
direct_shading_reconstructed = tf.nn.relu(direct_shading_reconstruction)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

sess.run(init)

saver.restore(sess, './checkpoints/shadingNet-440000')

imageFilePath = './iiw_110146.png'
ext = imageFilePath.rfind('.')

albedoFile =  imageFilePath[:ext] + '_final_albedo.png'
shadingFile = imageFilePath[:ext] + '_final_shading.png'
directShadingFile = imageFilePath[:ext] + '_final_directShading.png'
shadowFile = imageFilePath[:ext] + '_final_shadow.png'
ambientFile = imageFilePath[:ext] + '_final_ambient.png'
albedoFromAmbientFile = imageFilePath[:ext] + '_albedo_from_ambient.png'
albedoFromShadowFile = imageFilePath[:ext] + '_albedo_from_shadow.png'
albedoFromShadingFile = imageFilePath[:ext] + '_albedo_from_shading.png'

image = skimage.io.imread(imageFilePath).astype('float32')
image = cv2.resize(image, (image_height, image_width))

batch_composite = image[None, ...]
feed_dict = {I : batch_composite}


predictions_ambient_albedo, predictions_shadow_albedo, predictions_shading_albedo, predictions_ambient, predictions_shadow, predictions_direct_shading, predictions_albedo, predictions_shading = sess.run([ambient_albedo, shadow_albedo, shading_albedo, ambient_reconstructed, shadow_reconstructed, direct_shading_reconstructed, albedo_reconstructed, shading_reconstructed], feed_dict)

skimage.io.imsave(albedoFile, np.squeeze(predictions_albedo[0,...]).astype('uint8'))
skimage.io.imsave(shadingFile, np.squeeze(predictions_shading[0,...]).astype('uint8'))
skimage.io.imsave(directShadingFile, np.squeeze(predictions_direct_shading[0,...]).astype('uint8'))
skimage.io.imsave(shadowFile, np.squeeze(predictions_shadow[0,...]).astype('uint8'))
skimage.io.imsave(ambientFile, np.squeeze(predictions_ambient[0,...]).astype('uint8'))
skimage.io.imsave(albedoFromAmbientFile, np.squeeze(predictions_ambient_albedo[0,...]).astype('uint8'))
skimage.io.imsave(albedoFromShadowFile, np.squeeze(predictions_shadow_albedo[0,...]).astype('uint8'))
skimage.io.imsave(albedoFromShadingFile, np.squeeze(predictions_shading_albedo[0,...]).astype('uint8'))
