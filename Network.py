from __future__ import division
import tensorflow as tf
import sys
import tensorflow.contrib.slim as slim
sys.path.append("./deeplab")
from  deeplab.model import  _get_logits
import common



# USEFUL LAYERS
winit = tf.contrib.layers.xavier_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer



#https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/DeepLabV3_plus.py



def deeplabv3_plus(input_x=None, n_classes=20, training=True, height=512, width=512, batch_size=4):
    outputs_to_num_classes = {}
    outputs_to_num_classes['semantic'] = n_classes
    model_options = common.ModelOptions(outputs_to_num_classes=outputs_to_num_classes, crop_size=[height, width],
                                        atrous_rates=[6, 12, 18], output_stride=16, decoder_output_stride=4)

    output = _get_logits(input_x, model_options, is_training=training, fine_tune_batch_norm=True,
                    batch_size=batch_size)['semantic']

    output = tf.image.resize_bilinear(output, tf.shape(input_x)[1:3], align_corners=True)
    return output



def deeplabv3(input_x=None, n_classes=20, training=True, height=512, width=512, batch_size=4):
    outputs_to_num_classes = {}
    outputs_to_num_classes['semantic'] = n_classes
    model_options = common.ModelOptions(outputs_to_num_classes=outputs_to_num_classes, crop_size=[height, width],
                                        atrous_rates=[6, 12, 18], output_stride=16, decoder_output_stride=None)

    output = _get_logits(input_x, model_options, is_training=training, fine_tune_batch_norm=True,
                    batch_size=batch_size)['semantic']

    output = tf.image.resize_bilinear(output, tf.shape(input_x)[1:3], align_corners=True)
    return output

def deeplabv3_plus_plus(input_x=None, n_classes=20, training=True, height=512, width=512, batch_size=4):
    outputs_to_num_classes = {}
    outputs_to_num_classes['semantic'] = n_classes
    model_options = common.ModelOptions(outputs_to_num_classes=outputs_to_num_classes, crop_size=[height, width],
                                        atrous_rates=[6, 12, 18], output_stride=16)

    output = _get_logits(input_x, model_options, is_training=training, fine_tune_batch_norm=True,
                    batch_size=batch_size)['semantic']


    skip_connection_1 = tf.get_default_graph().get_tensor_by_name(
        "xception_65/entry_flow/block1/unit_1/xception_module/Relu:0")
    skip_connection_2 = tf.get_default_graph().get_tensor_by_name(
        "xception_65/entry_flow/block2/unit_1/xception_module/Relu:0")
    skip_connection_3 = tf.get_default_graph().get_tensor_by_name(
        "xception_65/entry_flow/block3/unit_1/xception_module/Relu:0")
    skip_connection_4 = tf.get_default_graph().get_tensor_by_name(
        "xception_65/middle_flow/block1/unit_1/xception_module/Relu:0")

    output_aspp = tf.get_default_graph().get_tensor_by_name(
        "concat_projection/Relu:0")

    #conv2d_sep(x, filters, filter_size, padding='same', strides=(1, 1), dilation_rate=(1, 1), activation='relu',  training=True, last=False)

    x = tf.image.resize_bilinear(output_aspp, (int(height/8), int(width/8)), align_corners=True)


    x_c = tf.concat([skip_connection_3, x], axis=3)
    x_c = conv2d_sep(x_c, 256, (3, 3),  dilation_rate=(1, 1), training=training)

    x = block_sep_conv(x_c, 256, (3, 3),  dilation_rate=(1, 1), training=training)
    x = block_sep_conv(x, 256, (3, 3),  dilation_rate=(2, 2), training=training)
    x = block_sep_conv(x, 256, (3, 3),  dilation_rate=(2, 2), training=training)
    x = block_sep_conv(x, 256, (3, 3),  dilation_rate=(4, 4), training=training)
    x = block_sep_conv(x, 256, (3, 3),  dilation_rate=(4, 4), training=training)
    x = block_sep_conv(x, 256, (3, 3),  dilation_rate=(1, 1), training=training)
    x = block_sep_conv(x, 256, (3, 3),  dilation_rate=(1, 1), training=training)
    x = block_sep_conv(x, 256, (3, 3),  dilation_rate=(1, 1), training=training)
    x = block_sep_conv(x, 256, (3, 3),  dilation_rate=(1, 1), training=training)
    x = tf.image.resize_bilinear(x, (int(height/4), int(width/4)), align_corners=True)
    # x = deconv2d_bn(x, 256, (3, 3),  training=training)

    x_c = tf.concat([skip_connection_2, x], axis=3)
    x_c = conv2d_sep(x_c, 128, (3, 3),  dilation_rate=(1, 1), training=training)


    x = block_sep_conv(x_c, 128, (3, 3), dilation_rate=(1, 1), training=training)
    x = block_sep_conv(x, 128, (3, 3), dilation_rate=(2, 2), training=training)
    x = block_sep_conv(x, 128, (3, 3), dilation_rate=(2, 2), training=training)
    x = block_sep_conv(x, 128, (3, 3), dilation_rate=(2, 2), training=training)
    x = block_sep_conv(x, 128, (3, 3), dilation_rate=(1, 1), training=training)
    x = block_sep_conv(x, 128, (3, 3), dilation_rate=(1, 1), training=training)
    x = block_sep_conv(x, 128, (3, 3), dilation_rate=(1, 1), training=training)
    x = tf.image.resize_bilinear(x, (int(height/2), int(width/2)), align_corners=True)
    # x = deconv2d_bn(x, 128, (3, 3),  training=training)

    x_c = tf.concat([skip_connection_1, x], axis=3)
    x_c = conv2d_sep(x_c, 64, (3, 3),  dilation_rate=(1, 1), training=training)
    x = block_sep_conv(x_c, 64, (3, 3), dilation_rate=(1, 1), training=training)



    #el kernel de la ultima que sea 1x1
    x = conv2d(x, n_classes, (1, 1), dilation_rate=(1, 1), training=training, last=True)
    output = tf.image.resize_bilinear(x, tf.shape(input_x)[1:3], align_corners=True)
    return output


def block_sep_conv(x, filters, filter_size, padding='same', strides=(1, 1), dilation_rate=(1, 1), activation='relu', training=True, last =False):
    x1 = conv2d_sep(x, filters, filter_size, padding, strides, dilation_rate, activation, training, last)
    x2 = conv2d_sep(x1, filters, filter_size, padding, strides, dilation_rate, activation, training, last)
    x3 = conv2d_sep(x2, filters, filter_size, padding, strides, dilation_rate, activation, training, last)
    return x+x3

def conv2d_sep(x, filters, filter_size, padding='same', strides=(1, 1), dilation_rate=(1, 1), activation='relu', training=True, last =False):

    with tf.name_scope('sep_conv'):
        x = tf.layers.separable_conv2d(x, filters, filter_size, strides=strides, padding=padding, use_bias=last, dilation_rate=dilation_rate,
        	activation=None, depthwise_initializer=winit, pointwise_initializer=winit,
                                       depthwise_regularizer=l2_regularizer(0.0002), pointwise_regularizer=l2_regularizer(0.0002))

        if not last:
	        x = tf.layers.batch_normalization(x,  training=training) 
	        '''
	        Activation fucntion
	        '''
	        if 'prelu' in  activation:
	        	x = tf.keras.layers.PReLU(x)
	        elif 'leakyrelu' in activation:
	        	x = tf.nn.leaky_relu(x)
	        elif 'relu' in activation:
	        	x = tf.nn.relu(x)

        return x



def conv2d(x, filters, filter_size, padding='same', strides=(1, 1), dilation_rate=(1, 1), activation='relu', training=True, last =False):
    with tf.name_scope('conv'):
        x = tf.layers.conv2d(x, filters, filter_size, strides=strides, padding=padding, dilation_rate=dilation_rate,
        	activation=None, kernel_initializer=winit, use_bias=last, kernel_regularizer=l2_regularizer(0.0002))

        if not last:

	        x = tf.layers.batch_normalization(x,  training=training)

	        '''
	        Activation fucntion
	        '''
	        if 'prelu' in  activation:
	        	x = tf.keras.layers.PReLU(x)
	        elif 'leakyrelu' in activation:
	        	x = tf.nn.leaky_relu(x)
	        elif 'relu' in activation:
	        	x = tf.nn.relu(x)

        return x

def deconv2d_bn(x, filters, filter_size, padding='same', strides=(1, 1), training=True, activation='relu'):

    with tf.name_scope('deconv'):

        x = tf.layers.conv2d_transpose(x, filters, filter_size, strides=strides, padding=padding, 
        	kernel_initializer=winit, activation=None, use_bias=False, kernel_regularizer=l2_regularizer(0.0002))
        x = tf.layers.batch_normalization(x,  training=training) 
        '''
        Activation fucntion
        '''
        if 'prelu' in  activation:
        	x = tf.keras.layers.PReLU(x)
        elif 'leakyrelu' in activation:
        	x = tf.nn.leaky_relu(x)
        elif 'relu' in activation:
        	x = tf.nn.relu(x)


        return x



def encoder_decoder_example(input_x=None, n_classes=20, training=True):

    x1 = conv2d(input_x, 64, (3, 3), padding='same', strides=(2, 2), dilation_rate=(1, 1), training=training)
    x2 = block_sep_conv(x1, 64, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x2 = conv2d_sep(x2, 128, (3, 3),  dilation_rate=(1, 1), training=training)
    x3 = block_sep_conv(x2, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x4 = block_sep_conv(x3, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x5 = block_sep_conv(x4, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(2, 2), training=training)
    x6 = conv2d_sep(x5, 256, (3, 3), padding='same', strides=(2, 2), dilation_rate=(2, 2), training=training)

    x7 = block_sep_conv(x6, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(4, 4), training=training)
    x8 = block_sep_conv(x7, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(4, 4), training=training)
    x9 = block_sep_conv(x8, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(4, 4), training=training)
    x9 = conv2d_sep(x9, 256, (3, 3), padding='same', strides=(2, 2), dilation_rate=(4, 4), training=training)
    x10 = block_sep_conv(x9, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(8, 8), training=training)
    x11 = block_sep_conv(x10, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(8, 8), training=training)
    x12 = block_sep_conv(x11, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(8, 8), training=training)
    x12 = deconv2d_bn(x12, 256, (3, 3), padding='same', strides=(2, 2),  training=training)
    x13 = block_sep_conv(x12, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x14 = block_sep_conv(x13, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x14 = block_sep_conv(x14, 256, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x15 = deconv2d_bn(x14, 256, (3, 3), padding='same', strides=(2, 2),  training=training)
    x15 = conv2d_sep(x15, 128, (3, 3),  dilation_rate=(1, 1), training=training)

    x16 = block_sep_conv(x15, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x17 = block_sep_conv(x16, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)
    x17 = block_sep_conv(x17, 128, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training)

    x18 = deconv2d_bn(x17, 32, (3, 3), padding='same', strides=(2, 2),  training=training)
    x19 = conv2d(x18, n_classes, (3, 3), padding='same', strides=(1, 1), dilation_rate=(1, 1), training=training, last=True)

    return x19

