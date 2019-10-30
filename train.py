import tensorflow as tf
import random
import os
import argparse
import time
from utils.utils import get_parameters
from Loader import Loader
import Network
import math
import numpy as np
import sys
from wave_check import wave_check_mask_loss
import matplotlib.pyplot as plt
import cv2
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

random.seed(os.urandom(7))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='./Datasets/EilatMixx') 
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample", default=3)
parser.add_argument("--augmentation", help="Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=8e-4)
parser.add_argument("--min_lr", help="Initial learning rate", default=7e-6)
parser.add_argument("--max_batch_size", help="batch_size", default=4)
parser.add_argument("--n_classes", help="number of classes to classify", default=10)
parser.add_argument("--ignore_label", help="class to ignore", default=255)
parser.add_argument("--epochs", help="Number of epochs to train", default=1000)
parser.add_argument("--width", help="width", default=512) 
parser.add_argument("--height", help="height", default=512)
parser.add_argument("--save_model", help="save_model", default=1)
parser.add_argument("--checkpoint_path", help="checkpoint path", default='./models/eilatmixx/') 
parser.add_argument("--train", help="if true, train, if not, test", default=1)
args = parser.parse_args()



# Hyperparameter
init_learning_rate = float(args.init_lr)
power_lr = 0.9
min_learning_rate = float(args.min_lr)
augmentation = bool(int(args.augmentation))
save_model = bool(int(args.save_model))
train_or_test = bool(int(args.train))
max_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
n_classes = int(args.n_classes)
ignore_label = int(args.ignore_label)
height = int(args.height)
channels = int(args.dimensions)
checkpoint_path = args.checkpoint_path

n_gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)


loader = Loader(dataFolderPath=args.dataset, n_classes=n_classes, problemType='segmentation', width=width,
                height=height, ignore_label=ignore_label, median_frequency=0.12)
testing_samples = len(loader.image_test_list)
training_samples = len(loader.image_train_list)

# Placeholders
training_flag = tf.placeholder(tf.bool)
input_x = tf.placeholder(tf.float32, shape=[max_batch_size, height, width, channels], name='input')
batch_images_visualization = tf.reverse(input_x, axis=[-1])  # opencv rgb -bgr
label = tf.placeholder(tf.float32, shape=[max_batch_size, height, width, n_classes + 1],
                       name='output')  # the n_classes + 1 is for the ignore classes
mask_label = tf.placeholder(tf.float32, shape=[max_batch_size, height, width], name='mask')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

# Network
output = Network.deeplabv3_plus_plus(input_x, n_classes, training_flag, height, width, max_batch_size)

# Get shapes
shape_output = tf.shape(output)
label_shape = tf.shape(label)
mask_label_shape = tf.shape(mask_label)

# Apply check wave loss
mask_label = wave_check_mask_loss(output, label, mask_label, times_check='max')

predictions = tf.reshape(output, [shape_output[1] * shape_output[2] * shape_output[0], shape_output[3]])
labels = tf.reshape(label, [label_shape[2] * label_shape[1] * label_shape[0], label_shape[3]])
mask_labels = tf.reshape(mask_label, [mask_label_shape[1] * mask_label_shape[0] * mask_label_shape[2]])

# Last class is the ignore class
labels_ignore = labels[:, n_classes]
labels_real = labels[:, :n_classes]

# Cross entropy loss
cost = tf.losses.softmax_cross_entropy(labels_real, predictions, weights=mask_labels)

# Metrics
labels = tf.argmax(labels, 1)
predictions = tf.argmax(predictions, 1)

indices = tf.squeeze(tf.where(tf.less_equal(labels, n_classes - 1)))  # ignore all labels >= num_classes
labels = tf.cast(tf.gather(labels, indices), tf.int64)
predictions = tf.gather(predictions, indices)

correct_prediction = tf.cast(tf.equal(labels, predictions), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)
acc, acc_op = tf.metrics.accuracy(labels, predictions)
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(labels, predictions, n_classes)
iou, conf_mat = tf.metrics.mean_iou(labels, predictions, n_classes)
conf_matrix_all = tf.confusion_matrix(labels, predictions, num_classes=n_classes)

 # Different variables
restore_variables = [var for var in tf.global_variables()]
train_variables = [var for var in tf.global_variables() ]
stream_vars = [i for i in tf.local_variables() if
               'count' in i.name or 'confusion_matrix' in i.name or 'total' in i.name]


# Count parameters
get_parameters()

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # adamOptimizer does not need lr decay
train = slim.learning.create_train_op(cost, optimizer)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    updates = tf.group(*update_ops)
    cost = control_flow_ops.with_dependencies([updates], cost)

saver = tf.train.Saver(tf.global_variables())
restorer = tf.train.Saver(restore_variables)

if not os.path.exists(os.path.join(checkpoint_path, 'iou')):
    os.makedirs(os.path.join(checkpoint_path, 'iou'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # get checkpoint if there is one
    ckpt_iou = tf.train.get_checkpoint_state(os.path.join(checkpoint_path, 'iou'))
    if ckpt_iou and tf.train.checkpoint_exists(ckpt_iou.model_checkpoint_path):
        print('Loading model...')
        restorer.restore(sess, ckpt_iou.model_checkpoint_path)
        print('Model loaded')

    if train_or_test:

        # Start variables
        batch_size = int(max_batch_size)
        best_val_loss = float('Inf')
        best_iou = float('-Inf')

        # EPOCH  loop
        for epoch in range(total_epochs):
            # Calculate tvariables for the batch and inizialize others
            time_first = time.time()
            epoch_learning_rate = (init_learning_rate - min_learning_rate) * math.pow(1 - epoch / 1. / total_epochs,
                                                                                      power_lr) + min_learning_rate

            print ("epoch " + str(epoch + 1) + ", lr: " + str(epoch_learning_rate) + ", batch_size: " + str(batch_size))

            total_steps = int(training_samples / batch_size) + 1
            show_each_steps = int(total_steps / 5)
            loss_acum = 0.0
            test_times = 0

            # steps in every epoch
            for step in range(total_steps):
                # get training data
                batch_x, batch_y, batch_mask = loader.get_batch(size=batch_size, train=True,
                                                                augmenter='segmentation')  
                train_feed_dict = {
                    input_x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    mask_label: batch_mask,
                    training_flag: True
                }
                _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

                # show info
                if step % show_each_steps == 0:
                    train_accuracy = sess.run([accuracy], feed_dict=train_feed_dict)

                    print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)

                    x_test, y_test, mask_test = loader.get_batch(size=batch_size, train=False)
                    test_feed_dict = {
                        input_x: x_test,
                        label: y_test,
                        mask_label: mask_test,
                        learning_rate: 0,
                        training_flag: False
                    }
                    acc_update, miou_update, mean_acc_update, val_loss = sess.run(
                        [acc_op, conf_mat, mean_acc_op, cost], feed_dict=test_feed_dict)
                    acc_total, miou_total, mean_acc_total, matrix_conf = sess.run([acc, iou, mean_acc, conf_matrix_all],
                                                                                  feed_dict=test_feed_dict)

                    loss_acum = loss_acum + val_loss
                    test_times = test_times + 1


            print("TEST")
            print("Accuracy: " + str(acc_total))
            print("miou: " + str(miou_total))
            print("mean accuracy: " + str(mean_acc_total))
            print("loss: " + str(loss_acum / test_times))
            # initialize metric variables for next epoch
            sess.run(tf.variables_initializer(stream_vars))

            # save models
            if save_model and best_iou < miou_total:
                best_iou = miou_total
                saver.save(sess=sess, save_path=os.path.join(checkpoint_path, 'iou', 'model.ckpt'))
            if save_model and best_val_loss > loss_acum / testing_samples:
                best_val_loss = loss_acum / testing_samples
                saver.save(sess=sess, save_path=os.path.join(checkpoint_path, 'model.ckpt'))

            # show tiem to finish training
            time_second = time.time()
            epochs_left = total_epochs - epoch - 1
            segundos_per_epoch = time_second - time_first
            print(str(segundos_per_epoch * epochs_left) + ' seconds to end the training. Hours: ' + str(
                segundos_per_epoch * epochs_left / 3600.0))




    else:
        # TEST
        loss_acum = 0.0
        loader.index_test = 0
        for i in xrange(0, testing_samples // max_batch_size):


            x_test, y_test, mask_test = loader.get_batch(size=max_batch_size, train=False)
            test_feed_dict = {
                input_x: x_test,
                label: y_test,
                mask_label: mask_test,
                learning_rate: 0,
                training_flag: False
            }
            image_salida, acc_update, miou_update, mean_acc_update, val_loss = sess.run([output, acc_op, conf_mat, mean_acc_op, cost],
                                                                          feed_dict=test_feed_dict)
            acc_total, miou_total, mean_acc_total, matrix_conf = sess.run([acc, iou, mean_acc, conf_matrix_all],
                                                                          feed_dict=test_feed_dict)
            loss_acum = loss_acum + val_loss

            dataset_name = args.dataset.split('/')[-1]
            if not os.path.exists('output/'+dataset_name):
                os.makedirs('output/'+dataset_name)

            image_salida = np.argmax(image_salida, 3)
            for index_output in xrange(max_batch_size):
                index_loader = max_batch_size * i +index_output
                name_split = loader.image_test_list[index_loader].split('/')
                name = name_split[len(name_split) - 1].replace('.jpg', '.png').replace('.jpeg', '.png')
                cv2.imwrite('output/'+dataset_name+ '/' + name, image_salida[index_output])

            if i == 0:
                confusion_matrix_total = matrix_conf
            else:
                confusion_matrix_total = confusion_matrix_total + matrix_conf

        print("TEST")
        print("Accuracy: " + str(acc_total))
        print("miou: " + str(miou_total))
        print("mean accuracy: " + str(mean_acc_total))
        print("loss: " + str(loss_acum / testing_samples))
        real = np.sum(confusion_matrix_total, axis=1)
        non_zeros = np.count_nonzero(real)

        print('Total classes in the dataset: ' + str(len(real)))
        print('In the evaluation only this number of classes are tested: ' + str(non_zeros))

        sum_real = np.sum(confusion_matrix_total, axis=1)
        row = 0
        for row_sum in sum_real:
            if row_sum == 0:
                confusion_matrix_total = np.delete(confusion_matrix_total, row, 0)
                confusion_matrix_total = np.delete(confusion_matrix_total, row, 1)
            else:
                row = row + 1

        matrix_conf = np.array(confusion_matrix_total, dtype=np.float32)
        matrix_conf = matrix_conf / matrix_conf.sum(axis=1)[:, None]
        plt.imshow(matrix_conf)  # , cmap=plt.cm.RdYlBu
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Confusion matrix ')
        plt.colorbar()
        plt.savefig('conf_matrix.png')
        plt.show()