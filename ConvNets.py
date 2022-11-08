import tensorflow as tf
import Layers as l

def first_pipeline(composite, weights, isTraining):

	# rgb spatial context encoder
	conv0 = tf.nn.conv2d(composite, weights['conv0'], strides=[1, 1, 1, 1], padding='SAME') # 256
	conv1_res1_base = l.res(conv0, weights['conv1_res1_1_base'], weights['conv1_res1_2_base'], isTraining)
	conv1_res2_base = l.res(conv1_res1_base, weights['conv1_res2_1_base'], weights['conv1_res2_2_base'], isTraining)
	conv1_res3_base = l.res(conv1_res2_base, weights['conv1_res3_1_base'], weights['conv1_res3_2_base'], isTraining)
	conv1_res4_base = l.res(conv1_res3_base, weights['conv1_res4_1_base'], weights['conv1_res4_2_base'], isTraining)
	conv1 = tf.nn.conv2d(conv1_res4_base, weights['conv1'], strides=[1, 2, 2, 1], padding='SAME') # 128
	conv1_res1 = l.res(conv1, weights['conv1_res1_1'], weights['conv1_res1_2'], isTraining)
	conv1_res2 = l.res(conv1_res1, weights['conv1_res2_1'], weights['conv1_res2_2'], isTraining)
	conv1_res3 = l.res(conv1_res2, weights['conv1_res3_1'], weights['conv1_res3_2'], isTraining)
	conv1_res4 = l.res(conv1_res3, weights['conv1_res4_1'], weights['conv1_res4_2'], isTraining)
	conv2 = tf.nn.conv2d(conv1_res4, weights['conv2'], strides=[1, 2, 2, 1], padding='SAME') # 64
	conv2_res1 = l.res(conv2, weights['conv2_res1_1'], weights['conv2_res1_2'], isTraining)
	conv2_res2 = l.res(conv2_res1, weights['conv2_res2_1'], weights['conv2_res2_2'], isTraining)
	conv2_res3 = l.res(conv2_res2, weights['conv2_res3_1'], weights['conv2_res3_2'], isTraining)
	conv2_res4 = l.res(conv2_res3, weights['conv2_res4_1'], weights['conv2_res4_2'], isTraining)
	conv3 = tf.nn.conv2d(conv2_res4, weights['conv3'], strides=[1, 2, 2, 1], padding='SAME') # 32
	conv3_res1 = l.res(conv3, weights['conv3_res1_1'], weights['conv3_res1_2'], isTraining)
	conv3_res2 = l.res(conv3_res1, weights['conv3_res2_1'], weights['conv3_res2_2'], isTraining)
	conv3_res3 = l.res(conv3_res2, weights['conv3_res3_1'], weights['conv3_res3_2'], isTraining)
	conv3_res4 = l.res(conv3_res3, weights['conv3_res4_1'], weights['conv3_res4_2'], isTraining)
	conv4 = tf.nn.conv2d(conv3_res4, weights['conv4'], strides=[1, 2, 2, 1], padding='SAME') # 16
	conv4_res1 = l.res(conv4, weights['conv4_res1_1'], weights['conv4_res1_2'], isTraining)
	conv4_res2 = l.res(conv4_res1, weights['conv4_res2_1'], weights['conv4_res2_2'], isTraining)
	conv4_res3 = l.res(conv4_res2, weights['conv4_res3_1'], weights['conv4_res3_2'], isTraining)
	conv4_res4 = l.res(conv4_res3, weights['conv4_res4_1'], weights['conv4_res4_2'], isTraining)
	conv5 = tf.nn.conv2d(conv4_res4, weights['conv5'], strides=[1, 2, 2, 1], padding='SAME') # 8
	#
	conv5_ambient = l.efficient_channel_attention(conv5, weights['bottleneck_attention_ambient'])
	conv5_shadow = l.efficient_channel_attention(conv5, weights['bottleneck_attention_shadow'])
	conv5_shading = l.efficient_channel_attention(conv5, weights['bottleneck_attention_shading'])
	# decoder albedo & ambient
	up_albedo_j = tf.image.resize_bilinear(conv5_ambient, (16, 16))
	deconv1_j = l.conv2d(up_albedo_j, weights['deconv1_j'], isTraining)
	up1_j = tf.image.resize_bilinear(deconv1_j, (32, 32))
	concat1_j = tf.concat([up1_j, conv3_res4], 3)
	deconv2_j = l.conv2d(concat1_j, weights['deconv2_j'], isTraining)
	up2_j = tf.image.resize_bilinear(deconv2_j, (64, 64))
	concat2_j = tf.concat([up2_j, conv2_res4], 3)
	deconv3_j = l.conv2d(concat2_j, weights['deconv3_j'], isTraining)
	up3_j = tf.image.resize_bilinear(deconv3_j, (128, 128))
	concat3_j = tf.concat([up3_j, conv1_res4], 3)
	deconv4_j = l.conv2d(concat3_j, weights['deconv4_j'], isTraining)
	up4_j = tf.image.resize_bilinear(deconv4_j, (256, 256))
	albedo_j = tf.nn.relu(tf.nn.conv2d(up4_j, weights['albedo_j'], strides=[1, 1, 1, 1], padding='SAME'))
	ambient = tf.nn.relu(tf.nn.conv2d(up4_j, weights['ambient'], strides=[1, 1, 1, 1], padding='SAME'))
	# decoder albedo & shadow
	up_albedo_k = tf.image.resize_bilinear(conv5_shadow, (16, 16))
	deconv1_k = l.conv2d(up_albedo_k, weights['deconv1_k'], isTraining)
	up1_k = tf.image.resize_bilinear(deconv1_k, (32, 32))
	concat1_k = tf.concat([up1_k, conv3_res4], 3)
	deconv2_k = l.conv2d(concat1_k, weights['deconv2_k'], isTraining)
	up2_k = tf.image.resize_bilinear(deconv2_k, (64, 64))
	concat2_k = tf.concat([up2_k, conv2_res4], 3)
	deconv3_k = l.conv2d(concat2_k, weights['deconv3_k'], isTraining)
	up3_k = tf.image.resize_bilinear(deconv3_k, (128, 128))
	concat3_k = tf.concat([up3_k, conv1_res4], 3)
	deconv4_k = l.conv2d(concat3_k, weights['deconv4_k'], isTraining)
	up4_k = tf.image.resize_bilinear(deconv4_k, (256, 256))
	albedo_k = tf.nn.relu(tf.nn.conv2d(up4_k, weights['albedo_k'], strides=[1, 1, 1, 1], padding='SAME'))
	shadow = tf.nn.relu(tf.nn.conv2d(up4_k, weights['shadow'], strides=[1, 1, 1, 1], padding='SAME'))
	# decoder albedo & shading
	up_albedo_p = tf.image.resize_bilinear(conv5_shading, (16, 16))
	deconv1_p = l.conv2d(up_albedo_p, weights['deconv1_p'], isTraining)
	up1_p = tf.image.resize_bilinear(deconv1_p, (32, 32))
	concat1_p = tf.concat([up1_p, conv3_res4], 3)
	deconv2_p = l.conv2d(concat1_p, weights['deconv2_p'], isTraining)
	up2_p = tf.image.resize_bilinear(deconv2_p, (64, 64))
	concat2_p = tf.concat([up2_p, conv2_res4], 3)
	deconv3_p = l.conv2d(concat2_p, weights['deconv3_p'], isTraining)
	up3_p = tf.image.resize_bilinear(deconv3_p, (128, 128))
	concat3_p = tf.concat([up3_p, conv1_res4], 3)
	deconv4_p = l.conv2d(concat3_p, weights['deconv4_p'], isTraining)
	up4_p = tf.image.resize_bilinear(deconv4_p, (256, 256))
	albedo_p = tf.nn.relu(tf.nn.conv2d(up4_p, weights['albedo_p'], strides=[1, 1, 1, 1], padding='SAME'))
	shading = tf.nn.relu(tf.nn.conv2d(up4_p, weights['shading'], strides=[1, 1, 1, 1], padding='SAME'))
	# fusion
	albedo_input = tf.concat([albedo_j, albedo_k, albedo_p], 3)

	fusion = tf.nn.conv2d(albedo_input, weights['fuse'], strides=[1, 1, 1, 1], padding='SAME') # fusion with 1x1 conv for weighted average

	fusion_input = tf.concat([albedo_input, fusion, shadow, ambient], 3) 

	fuse_conv_1 = tf.nn.conv2d(fusion_input, weights['fuse_conv_1'], strides=[1, 1, 1, 1], padding='SAME') # 128
	fuse_res1 = l.dilated_res(fuse_conv_1, weights['fuse_res1_1'], weights['fuse_res1_2'], 2, isTraining)
	fuse_res2 = l.dilated_res(fuse_res1, weights['fuse_res2_1'], weights['fuse_res2_2'], 2, isTraining)
	fuse_res3 = l.dilated_res(fuse_res2, weights['fuse_res3_1'], weights['fuse_res3_2'], 4, isTraining)
	fuse_res4 = l.dilated_res(fuse_res3, weights['fuse_res4_1'], weights['fuse_res4_2'], 8, isTraining)
	fuse_res5 = l.dilated_res(fuse_res4, weights['fuse_res5_1'], weights['fuse_res5_2'], 8, isTraining)
	fuse_res6 = l.res(fuse_res5, weights['fuse_res6_1'], weights['fuse_res6_2'], isTraining)

	albedo = tf.nn.relu(tf.nn.conv2d(fuse_res6, weights['albedo'], strides=[1, 1, 1, 1], padding='SAME'))

	return albedo_j, albedo_k, albedo_p, albedo, shading, ambient, shadow
