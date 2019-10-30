import tensorflow as tf
import random
import os
import math
import numpy as np

random.seed(os.urandom(7))
from scipy import signal


def wave_check_mask_loss(output, label, mask_label, times_check=8):
    assert times_check > 0, "times_check must be greater than 0"

    # Get shapes
    output_arg_max = tf.argmax(output, 3)
    labels_arg_max = tf.argmax(label, 3)
    factor_loss = np.ones((labels_arg_max.shape))

    # calculate max distance to check
    max_distance = float(max(labels_arg_max.shape[1].value, labels_arg_max.shape[2].value))
    times_check_maximum = int(math.log(max_distance, 2))

    if str(times_check) in 'max':
        times_check = int(times_check_maximum - 1)

    assert times_check < times_check_maximum, "times_check must be smaller than " + str(times_check_maximum)

    multiply_factor = signal.gaussian((times_check_maximum - 1) * 2, std=3)[
                      int(times_check_maximum - 1):int(times_check_maximum - 1 + times_check)]

    pixels_distance_list = []
    for i in xrange(times_check):
        value = math.pow(2, i)
        pixels_distance_list = pixels_distance_list + [int(round(value))]

    for i, times in zip(pixels_distance_list, xrange(times_check)):
        output_arg_max_shifted_left = tf.manip.roll(output_arg_max, shift=[0, i], axis=[1, 2])
        output_arg_max_shifted_up = tf.manip.roll(output_arg_max, shift=[i, 0], axis=[1, 2])
        output_arg_max_shifted_down = tf.manip.roll(output_arg_max, shift=[-i, 0], axis=[1, 2])
        output_arg_max_shifted_right = tf.manip.roll(output_arg_max, shift=[0, -i], axis=[1, 2])

        predicted_not_same_as_down = tf.cast(tf.logical_not(tf.equal(output_arg_max_shifted_up, output_arg_max)), tf.float32)
        predicted_not_same_as_left = tf.cast(tf.logical_not(tf.equal(output_arg_max_shifted_right, output_arg_max)), tf.float32)
        predicted_not_same_as_right = tf.cast(tf.logical_not(tf.equal(output_arg_max_shifted_left, output_arg_max)), tf.float32)
        predicted_not_same_as_up = tf.cast(tf.logical_not(tf.equal(output_arg_max_shifted_down, output_arg_max)), tf.float32)

        ones_mask = np.ones((labels_arg_max.shape))

        mask_left = ones_mask.copy()
        mask_left[:, :, -i:] = 0
        mask_rigth = ones_mask.copy()
        mask_rigth[:, :, :i] = 0
        mask_down = ones_mask.copy()
        mask_down[:, :i, :] = 0
        mask_up = ones_mask.copy()
        mask_up[:, -i:, :] = 0

        predicted_not_same_as_left = predicted_not_same_as_left * mask_left
        predicted_not_same_as_right = predicted_not_same_as_right * mask_rigth
        predicted_not_same_as_up = predicted_not_same_as_up * mask_up
        predicted_not_same_as_down = predicted_not_same_as_down * mask_down

        factor_loss += (
                       predicted_not_same_as_left + predicted_not_same_as_right + predicted_not_same_as_up + predicted_not_same_as_down) * \
                       multiply_factor[times]

    # Normalize the factor_loss to have a pixel mean weight of 1
    mean_factor = tf.reduce_mean(factor_loss, axis=[1, 2])
    mean_factor = tf.reshape(mean_factor, [labels_arg_max.shape[0], 1, 1])

    factor_loss = factor_loss * 1. / mean_factor

    return mask_label * factor_loss

