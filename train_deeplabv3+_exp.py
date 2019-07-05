import os
import time
import random
import numpy as np
import PIL.Image as Image
import pickle
import matplotlib.pyplot as plt
import csv
import re

import tensorflow as tf
from deeplab.core import feature_extractor
from deeplab import common
from deeplab import model
from deeplab.utils import train_utils
from tensorflow.keras import backend as K
import cv2
from PIL import Image
from scipy import ndimage

import ipdb


# gpunumber = 0
# os.environ["CUDA_VISIBLE_DEVICES"]= str(gpunumber)

flags = tf.app.flags

# Loading path
flags.DEFINE_boolean('use_external', False, 'using external dataset')
flags.DEFINE_string('data_path', 'dataset/', 'directory of dataset')
flags.DEFINE_string('feature', 'xception/', 'directory feature extractor')
flags.DEFINE_string('tf_initial_checkpoint', 'xception/xception71/model.ckpt',
                    'The initial checkpoint in tensorflow format.')
                    
#flags.DEFINE_boolean('use_external', False, 'using external dataset')
#flags.DEFINE_string('data_path', 'dataset/', 'directory of dataset')
#flags.DEFINE_string('feature', 'resnet_v1_50/', 'directory feature extractor')

flags.DEFINE_boolean('finetune', False, 'using external dataset')
flags.DEFINE_string('encoder_model', 'pretrained/encoder.ckpt-67500', 'directory of encoder model')
flags.DEFINE_string('decoder_model', 'pretrained/decoder.ckpt-67500', 'directory of decoder model')

# Saving directory
flags.DEFINE_string("experiment_name", "deeplabv3+", "name of the experiment directory.")
flags.DEFINE_string('save_path', '/mnt/sda/ganpizza/runs', 'saving directotry')
flags.DEFINE_boolean('save_soft_masks', False, 'Saving test results')
flags.DEFINE_boolean('save_csv_file', True, 'Saving submission file')

# Training parameters
flags.DEFINE_integer('batch_size', 2, 'Batch size.')
flags.DEFINE_integer('max_steps', 80000, 'Number of steps to run training.')
flags.DEFINE_float('f_lr', 3e-6, 'learning rate')
flags.DEFINE_float('d_lr', 3e-6, 'learning rate')
flags.DEFINE_float('weight_jac', 0.1, 'balance param')
flags.DEFINE_float('weight_tv', 0.0000001, 'balance param tv')
flags.DEFINE_string('loss', 'class', "choice between cross-entropy and focal loss.")
flags.DEFINE_float('scale_factor', 2.0, "random scaling factor.")

# Deeplab param

flags.DEFINE_multi_integer('train_crop_size', [400, 400],
                           'Image crop size [height, width] during training.')

flags.DEFINE_multi_integer('atrous_rates',  [6, 12, 18],
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

# Define the mode between train/val or final
flags.DEFINE_string("cil_mode", "final", "modes between trainval and final")
flags.DEFINE_string("mode", "train", "Set code mode between train/test and pure testing")

FLAGS = flags.FLAGS

save_log_path = None
save_model_path = None
save_output_path = None


def buildExternalDataset (data_path,
                          batch_size=16,
                          shuffle=True,
                          crop_size=[512, 512],
                          augment=False,
                          scale_factor=None,
                          buffer = 30):
    
    def _random_crop(image, label, size):
        combined = tf.concat([image, label], axis=2)
        last_label_dim = tf.shape(label)[-1]
        last_image_dim = tf.shape(image)[-1]
        combined_crop = tf.random_crop(combined,
            size=tf.concat([size, [last_label_dim + last_image_dim]], axis=0) )
        return combined_crop[:, :, :last_image_dim], combined_crop[:, :, last_image_dim:]

    def _random_crop_new(image, label, size):
        h, w, _ = image.shape

        # Perform random scaling
        if scale_factor is not None:
            # decide the random scaling factor
            random_scale = np.random.uniform(1., scale_factor)

            # Perform scaling
            h_new = int(h * random_scale)
            w_new = int(w * random_scale)

            image = cv2.resize(image, (h_new, w_new), cv2.INTER_LINEAR)
            label = cv2.resize(label, (h_new, w_new), cv2.INTER_NEAREST)[..., None]

        h, w, _ = image.shape

        # Define cropping range
        h_start = 0
        w_start = 0
        h_end = h - size[0]
        w_end = w - size[0]

        # keep cropping
        max_iter=4000
        min_ratio = 0.06

        iter = 0
        while iter < max_iter:
            h_crop = np.random.randint(h_start, h_end)
            w_crop = np.random.randint(w_start, w_end)
            cropped_image = image[h_crop:h_crop+size[0], w_crop:w_crop+size[1], :]
            cropped_label = label[h_crop:h_crop+size[0], w_crop:w_crop+size[1], :]

            if np.mean(cropped_label >= min_ratio):
                break
            iter += 1

        return cropped_image, cropped_label


    def _flip(image, label):
        combined = tf.concat([image, label], axis=2)
        last_label_dim = tf.shape(label)[-1]
        last_image_dim = tf.shape(image)[-1]

        combined_flip = tf.image.random_flip_left_right(combined, seed=0 )
        combined_flip = tf.image.random_flip_up_down(combined_flip, seed=0 )

        return combined_flip[:, :, :last_image_dim], combined_flip[:, :, last_image_dim:]

    def _rotate(image, label):
        combined = tf.concat([image, label], axis=2)
        last_label_dim = tf.shape(label)[-1]
        last_image_dim = tf.shape(image)[-1]

        combined_rot = tf.image.rot90(combined, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        return combined_rot[:, :, :last_image_dim], combined_rot[:, :, last_image_dim:]

    def _load_train_image(image_path, label_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        
        label = tf.io.read_file(label_path)
        label = tf.io.decode_jpeg(label)
        label = label / 200
        label = tf.math.round(label)
        label = tf.cast(label,tf.uint8)

        return image, label
    
    def _load_test_image(image_path, label_path, ids, size):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        image = tf.image.crop_to_bounding_box(image, 300,300,512,512)

        label = tf.io.read_file(label_path)
        label = tf.io.decode_jpeg(label)
        label = tf.image.crop_to_bounding_box(label, 300,300,512,512)
        label = label / 200
        label = tf.math.round(label)
        label = tf.cast(label,tf.uint8)

        return image, label, ids

    all_trian_image = os.listdir(data_path + 'new_RGB/')
    all_train_label = os.listdir(data_path + 'new_mask/')

    all_trian_image_paths = sorted([ data_path + 'new_RGB/' + str(path) for path in all_trian_image])
    all_trian_label_paths = sorted([ data_path + 'new_mask/' + str(path) for path in all_train_label])

    all_test_image = os.listdir(data_path + 'test_RGB/')
    all_test_label = os.listdir(data_path + 'test_mask/')

    all_test_image_paths = sorted([ data_path + 'test_RGB/' + str(path) for path in all_test_image])
    all_test_label_paths = sorted([ data_path + 'test_mask/' + str(path) for path in all_test_label])
    ids = [i for i in range(20)]

    train_dataset = tf.data.Dataset.from_tensor_slices((all_trian_image_paths, all_trian_label_paths)).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((all_test_image_paths, all_test_label_paths, ids))

    if shuffle:
        train_dataset = train_dataset.shuffle(len(all_trian_image_paths))

    train_dataset = train_dataset.map(_load_train_image).prefetch(buffer)

    if crop_size[0] is not None:
        train_dataset = train_dataset.map(lambda image, label: 
                        tuple(tf.py_func(_random_crop_new, [image, label, crop_size],
                        [tf.uint8, tf.uint8] )))

    if augment:
        train_dataset = train_dataset.map(lambda image, label:
                    tuple(tf.py_function(_flip, [image, label], [tf.uint8, tf.uint8])))
        train_dataset = train_dataset.map(lambda image, label:
                    tuple(tf.py_function(_rotate, [image, label], [tf.uint8, tf.uint8])))
        # ToDo: whether add color jitter

    train_dataset = train_dataset.batch(batch_size)

    test_dataset = test_dataset.map(lambda image, label, ids: 
                        tuple( tf.py_function( _load_test_image, [image, label, ids, crop_size],
                        [tf.uint8, tf.uint8, tf.int32] ))).prefetch(buffer)

    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset


def buildTestDataset ( data_path, batch_size = 16, buffer = 30):
    def _load_test_image(image_path, ids):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        return image, ids

    all_test_timage =  os.listdir(data_path + 'test_input/')
    all_test_image_paths = sorted([ data_path + 'test_input/' + str(path) for path in all_test_timage])

    ids = [int(re.search(r"\d+", path).group(0)) for path in all_test_image_paths]

    dataset = tf.data.Dataset.from_tensor_slices((all_test_image_paths, ids))

    dataset = dataset.map(_load_test_image).prefetch(buffer)

    dataset = dataset.batch(batch_size)


    return dataset

def buildValidDataset( data_path, batch_size = 16, buffer = 30):

    def _load_image_label(image_path, label_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        label = tf.io.read_file(label_path)
        label = tf.io.decode_png(label)
        label = label / 255
        label = tf.math.round(label)

        #label = tf.div(tf.io.decode_png(label), 255 )
        label = tf.cast(label,tf.uint8)

        return image, label

    if FLAGS.use_external is True:
        image_path = os.path.join(data_path, 'train_input/')
        label_path = os.path.join(data_path, 'train_output/')
    else:
        image_path = os.path.join(data_path, 'val_input/')
        label_path = os.path.join(data_path, 'val_output/')

    all_valid_image = os.listdir(image_path)
    all_valid_label = os.listdir(label_path)

    all_valid_image_paths = sorted([image_path + str(path) for path in all_valid_image])
    all_valid_label_paths = sorted([label_path + str(path) for path in all_valid_label])

    dataset = tf.data.Dataset.from_tensor_slices((all_valid_image_paths, all_valid_label_paths))

    dataset = dataset.map(_load_image_label).prefetch(buffer)

    dataset = dataset.batch(batch_size)

    return dataset


def buildTrainDataset(data_path, batch_size=16, shuffle=True,
                      crop_size=[400, 400],
                      augment=False,
                      buffer=30,
                      scale_factor=None,
                      mode="train_val"):
    def _load_image_label(image_path, label_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        label = tf.io.read_file(label_path)
        label = tf.io.decode_png(label)
        label = label / 255
        label = tf.math.round(label)

        #label = tf.div(tf.io.decode_png(label), 255 )
        label = tf.cast(label,tf.uint8)

        return image, label

    def _random_crop_new(image, label, size):
        h, w, _ = image.shape

        # Perform random scaling
        if scale_factor is not None:
            # decide the random scaling factor
            random_scale = np.random.uniform(1., scale_factor)

            # Perform scaling
            h_new = int(h * random_scale)
            w_new = int(w * random_scale)

            image = cv2.resize(image, (h_new, w_new), cv2.INTER_LINEAR)
            label = cv2.resize(label, (h_new, w_new), cv2.INTER_NEAREST)[..., None]

        h, w, _ = image.shape

        # Define cropping range
        h_start = 0
        w_start = 0
        h_end = max(h - size[0], 0)
        w_end = max(w - size[0], 0)

        # keep cropping
        max_iter=4000
        min_ratio = 0.06

        if (h_end > 0) and (w_end > 0):
            h_crop = np.random.randint(h_start, h_end)
            w_crop = np.random.randint(w_start, w_end)
        else:
            h_crop = 0
            w_crop = 0

        cropped_image = image[h_crop:h_crop+size[0], w_crop:w_crop+size[1], :]
        cropped_label = label[h_crop:h_crop+size[0], w_crop:w_crop+size[1], :]

        return cropped_image, cropped_label

    def _random_crop(image, label, size):
        combined = tf.concat([image, label], axis=2)
        last_label_dim = tf.shape(label)[-1]
        last_image_dim = tf.shape(image)[-1]
        combined_crop = tf.random_crop(combined,
            size=tf.concat([size, [last_label_dim + last_image_dim]], axis=0) )
        return combined_crop[:, :, :last_image_dim], combined_crop[:, :, last_image_dim:]

    def _flip(image, label):
        combined = tf.concat([image, label], axis=2)
        last_label_dim = tf.shape(label)[-1]
        last_image_dim = tf.shape(image)[-1]

        combined_flip = tf.image.random_flip_left_right(combined, seed=0 )
        combined_flip = tf.image.random_flip_up_down(combined_flip, seed=0 )

        return combined_flip[:, :, :last_image_dim], combined_flip[:, :, last_image_dim:]

    def _rotate(image, label):
        combined = tf.concat([image, label], axis=2)
        last_label_dim = tf.shape(label)[-1]
        last_image_dim = tf.shape(image)[-1]

        combined_rot = tf.image.rot90(combined, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        return combined_rot[:, :, :last_image_dim], combined_rot[:, :, last_image_dim:]

    # Get all train input file names
    train_image_path = os.path.join(data_path, 'train_input/')
    train_label_path = os.path.join(data_path, 'train_output/')
    all_trian_image = os.listdir(train_image_path)
    all_train_label = os.listdir(train_label_path)

    # Get all val input file names
    val_image_path = os.path.join(data_path, "val_input/")
    val_label_path = os.path.join(data_path, "val_output/")
    all_val_image = os.listdir(val_image_path)
    all_val_label = os.listdir(val_label_path)

    # Use different training set in different mode
    assert mode in ["train_val", "final"]
    if mode == "train_val":
        all_trian_image = list(set(all_trian_image) - set(all_val_image))
        all_train_label = list(set(all_train_label) - set(all_val_label))

    # ipdb.set_trace()
    all_trian_image_paths = sorted([train_image_path + str(path) for path in all_trian_image])
    all_trian_label_paths = sorted([train_label_path + str(path) for path in all_train_label])
    assert len(all_trian_image_paths) > 0
    assert len(all_trian_label_paths) > 0

    # ipdb.set_trace()

    dataset = tf.data.Dataset.from_tensor_slices((all_trian_image_paths, all_trian_label_paths)).repeat()

    dataset = dataset.map(_load_image_label).prefetch(buffer)

    if shuffle:
        dataset = dataset.shuffle(1000)

    # ipdb.set_trace()
    if crop_size[0] is not None:
        dataset = dataset.map(lambda image, label: 
                        tuple(tf.py_func(_random_crop_new, [image, label, crop_size],
                        [tf.uint8, tf.uint8])))

    if augment:
        dataset = dataset.map(lambda image, label:
                    tuple(tf.py_function(_flip, [image, label], [tf.uint8, tf.uint8])))
        dataset = dataset.map(lambda image, label:
                    tuple(tf.py_function(_rotate, [image, label], [tf.uint8, tf.uint8])))

    dataset = dataset.batch(batch_size)

    return dataset


def create_submission_files(test_predictions, test_ids, output_path, count):

    def patch_to_label(patch):
        # percentage of pixels > 1 required to assign a foreground label to a patch
        foreground_threshold = 0.5
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0
    file_path = os.path.join(output_path, 'submission_'+str(count)+'.csv' )

    with open(file_path, 'w') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["id", "prediction"])

        patch_size = 16
        for i, label in enumerate(test_predictions):
            im = test_predictions[i]
            img_number = test_ids[i]
            for j in range(0, im.shape[1], patch_size):
                for i in range(0, im.shape[0], patch_size):
                        patch = im[i:i + patch_size, j:j + patch_size]
                        label = patch_to_label(patch)
                        outcsv.writelines("{:03d}_{}_{},{}\n".format(img_number, j, i, label))


def create_soft_label_mask(soft_masks, output_path, count):

    file_path = os.path.join(output_path, 'softLabels_'+str(count)+'.pkl' )

    # print(len(soft_masks))
    # print(soft_masks[0].shape)

    with open(file_path, 'wb') as f:
        pickle.dump(soft_masks, f)


def create_soft_label_mask_ag(soft_masks, output_path, count):

    file_path = os.path.join(output_path, 'softLabels_ag_'+str(count)+'.pkl' )

    # print(len(soft_masks))
    # print(soft_masks[0].shape)

    with open(file_path, 'wb') as f:
        pickle.dump(soft_masks, f)


def create_valid_soft_label(soft_masks, output_path, count):
    file_path = os.path.join(output_path, 'valid_softLabels_'+str(count)+'.pkl' )

    # print(len(soft_masks))
    # print(soft_masks[0].shape)

    with open(file_path, 'wb') as f:
        pickle.dump(soft_masks, f)


def _variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def main():
    model_options1 = common.ModelOptions(
                    outputs_to_num_classes=2,
                    crop_size=None,
                    atrous_rates=FLAGS.atrous_rates,
                    output_stride=FLAGS.output_stride)


    with tf.name_scope('input_placeholder'):
        input_placeholder = tf.placeholder(tf.int32, shape=(None, None, None, 3 ), name = 'inputs')

    with tf.name_scope('label_placeholder'):
        label_placeholder = tf.placeholder(tf.int32, shape=(None, None, None, 1), name = 'labels')

    with tf.name_scope('test_image'):
        test_rgb_placeholder = tf.placeholder(tf.int32, shape=(None, None, None, 3), name = 'image_rgb')
        test_mask_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1), name = 'image_mask')

    with tf.variable_scope('feature_extractor') as feature_vs:


        feature_3, train_endpoint = model.extract_features(input_placeholder, model_options1, is_training=True)

        feature_1 = train_endpoint['feature_extractor/xception_71/entry_flow/block3/unit_1/xception_module/'
                                   'separable_conv2_pointwise']
        feature_2 = train_endpoint['feature_extractor/xception_71/entry_flow/block4/unit_1/xception_module/'
                                   'separable_conv2_pointwise']

        feature_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'feature_extractor')

    # ipdb.set_trace()
    def Decode(feature_map_1, feature_map_2, feature, scope):
        
        with tf.variable_scope(scope) as decoder_vs:
            # ipdb.set_trace()
            conv_map1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same')(feature_map_1)
 
            #up_map2 = tf.keras.layers.UpSampling2D(size=(2, 2))(feature_map_2)
            up_map2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature_map_2)

            concat_map1 = tf.concat([conv_map1, up_map2], axis=3)
        
            #up_map3 = tf.keras.layers.UpSampling2D(size=(4, 4))(feature)
            up_map3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature)

            conv_map2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(concat_map1)
            
            concat_map2 = tf.concat([conv_map2, up_map3], axis=3)

            conv_map3 = tf.keras.layers.Conv2D(filters=2048, kernel_size=(3, 3), padding='same')(concat_map2)

            prediction = tf.keras.layers.Conv2D(2, (1, 1), kernel_initializer='he_normal',
                        activation='linear', padding='same', strides=(1, 1))(conv_map3)
        
            #prediction = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_map3)
            logits = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(prediction)

            # ipdb.set_trace()
        return logits

    with tf.name_scope('decoder_fn'):
        pred_soft = Decode(feature_1, feature_2, feature_3, scope = 'decoder')
        pred_hard = tf.expand_dims(tf.argmax (pred_soft, axis=3), -1 )
        soft_label = tf.nn.softmax(pred_soft, axis=3)

        decoder_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'decoder')

    with tf.name_scope('loss'):
        # Standard cross entropy loss
        with tf.name_scope("classifiaction_loss"):
            class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= pred_soft, labels= tf.squeeze(label_placeholder, axis=3))
            class_loss = tf.reduce_mean(class_loss)

        # Focal-loss
        with tf.name_scope("focal_loss"):
            gamma = 0.5
            eps = 1e-7
            onehot_label = tf.one_hot(tf.squeeze(label_placeholder), depth=2, axis=-1)
            pred_softmax = tf.clip_by_value(tf.nn.softmax(pred_soft, axis=3), eps, 1-eps)
            focal_loss = -1 * onehot_label * tf.log(pred_softmax)
            focal_loss = tf.reduce_mean(focal_loss * ((1 - pred_softmax) ** gamma))

        # Weighted cross-entropy loss
        with tf.name_scope("weighted_class_loss"):
            label = tf.one_hot(tf.squeeze(label_placeholder), depth=2, axis=-1)
            wclass_loss = tf.nn.weighted_cross_entropy_with_logits(logits=pred_soft, targets=label,
                                                                   pos_weight=1.1)
            wclass_loss = tf.reduce_mean(wclass_loss)

        # Smoothness loss
        # with tf.name_scope("smoothness_loss"):
        '''
        gt_onehot = tf.one_hot(tf.squeeze(label_placeholder, axis=3), depth = 2, dtype = tf.float32)
        smooth = 1e-9
        intersection = tf.reduce_sum(soft_label * gt_onehot)
        union = tf.reduce_sum(soft_label) + tf.reduce_sum(gt_onehot)
        jaccard_index = (intersection + smooth) / ( union - intersection + smooth) 
        jaccard_loss =  (1 - jaccard_index) 
        '''
        # Define the jacard loss
        with tf.name_scope("jacard_loss"):
            smooth = 1e-9
            y_head = tf.slice(soft_label, [0, 0, 0, 1], [-1, -1, -1,-1])

            yy_head = tf.reduce_sum(tf.multiply(y_head, tf.cast(label_placeholder, tf.float32)))

            jaccard_index = (yy_head + smooth) / ( tf.reduce_sum(tf.cast(label_placeholder, tf.float32)) + tf.reduce_sum(y_head) - yy_head + smooth)
            jaccard_loss = (1 - jaccard_index)

        # Define the total variation loss
        with tf.name_scope("tv_loss"):
            tv_loss = tf.reduce_mean(tf.image.total_variation(soft_label))

        if FLAGS.loss == "focal":
            total_loss = 1.0 * focal_loss + FLAGS.weight_jac * jaccard_loss + FLAGS.weight_tv * tv_loss
            tf.summary.scalar('class_loss', focal_loss)

        elif FLAGS.loss == "wclass":
            total_loss = 1.0 * wclass_loss + FLAGS.weight_jac * jaccard_loss + FLAGS.weight_tv * tv_loss
            tf.summary.scalar('class_loss', wclass_loss)
        else:
            total_loss = 1.0 * class_loss + FLAGS.weight_jac * jaccard_loss + FLAGS.weight_tv * tv_loss
            tf.summary.scalar('class_loss', class_loss)

        tf.summary.scalar('jaccard_loss', jaccard_loss)
        tf.summary.scalar('tv_loss', tv_loss)
        tf.summary.scalar('total_loss', total_loss) 

    with tf.name_scope('input_image'):
        tf.summary.image('input', tf.cast(input_placeholder, tf.uint8))

    with tf.name_scope('prediction_image'):
        tf.summary.image('prediction', tf.cast(pred_hard, tf.float32))

    with tf.name_scope('groundtruth'):
        tf.summary.image('groundtruth', tf.cast(label_placeholder, tf.float32))


    with tf.name_scope('test_rgb'):
        tf.summary.image('test_rgb', tf.cast(test_rgb_placeholder, tf.uint8), max_outputs=10)
    
    with tf.name_scope('test_mask'):
        tf.summary.image('test_mask', tf.cast(test_mask_placeholder, tf.float32), max_outputs=10)


    with tf.name_scope('summary_var'):
        _variable_summaries(decoder_varlist[0])
        _variable_summaries(decoder_varlist[1])

    with tf.name_scope('init_function'):

        last_layers = model.get_extra_layer_scopes(FLAGS.last_layers_contain_logits_only)
        scaffold = tf.train.Scaffold()

        variables_to_restore = tf.contrib.framework.get_variables_to_restore()

        # ipdb.set_trace()
        init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
                        FLAGS.tf_initial_checkpoint,
                        variables_to_restore,
                        ignore_missing_vars=True)

    with tf.name_scope('optimizer'):
        global_step = tf.train.get_or_create_global_step()
        if FLAGS.use_external == True:
            decay_steps = FLAGS.max_steps
        else:
            decay_steps = 10000

        f_lr = tf.train.polynomial_decay(FLAGS.f_lr, global_step,
                                         decay_steps=decay_steps,
                                         end_learning_rate=FLAGS.f_lr * 0.01)
        d_lr = tf.train.polynomial_decay(FLAGS.d_lr, global_step,
                                         decay_steps=decay_steps,
                                         end_learning_rate=FLAGS.f_lr * 0.01)
        feature_opt = tf.train.AdamOptimizer(f_lr).minimize(total_loss, var_list = feature_varlist,
                                                            global_step=global_step)
        decoder_opt = tf.train.AdamOptimizer(d_lr).minimize(total_loss, var_list = decoder_varlist)

    with tf.name_scope("lr_summary"):
        tf.summary.scalar("f_lr", f_lr, family="lr")
        tf.summary.scalar("d_lr", d_lr, family="lr")

    init_var = tf.global_variables_initializer()

    with tf.name_scope('saver'):
        feature_saver = tf.train.Saver(feature_varlist, max_to_keep=10)
        decoder_saver = tf.train.Saver(decoder_varlist, max_to_keep=10)

    # ipdb.set_trace()
    with tf.name_scope('dataset'):
        # ipdb.set_trace()
        if (FLAGS.use_external):
            data_path = os.path.join(FLAGS.data_path, "external/")
            train_data, test_data = buildExternalDataset(data_path, batch_size = FLAGS.batch_size,
                                        shuffle=True, crop_size=[400, 400],
                                        augment=True, buffer=100, scale_factor=FLAGS.scale_factor)

            # ToDo [julin] use cil dataset as validation set
            # Actually, we use the cil dataset as the validation dataset in external mode
            valid_data = buildValidDataset(FLAGS.data_path, batch_size=FLAGS.batch_size)
        else:
            train_data = buildTrainDataset(FLAGS.data_path,
                                           batch_size = FLAGS.batch_size,
                                           shuffle=True,
                                           crop_size=[400, 400],
                                           augment=True,
                                           scale_factor=FLAGS.scale_factor,
                                           mode=FLAGS.cil_mode)
            test_data = buildTestDataset(FLAGS.data_path,
                                         batch_size=FLAGS.batch_size)

            valid_data = buildValidDataset(FLAGS.data_path,
                                           batch_size=FLAGS.batch_size)



        with tf.name_scope('dataset_iterator'):
            it = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            next_data = it.get_next()
            init_data = it.make_initializer(train_data)
            
            test_iterator = tf.data.Iterator.from_structure(test_data.output_types, test_data.output_shapes)
            next_test = test_iterator.get_next()
            init_test = test_iterator.make_initializer(test_data)

            # if not FLAGS.use_external:
            valid_iterator = tf.data.Iterator.from_structure(valid_data.output_types, valid_data.output_shapes)
            next_valid = valid_iterator.get_next()
            init_valid = valid_iterator.make_initializer(valid_data)

        # ipdb.set_trace()

    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session( config=config )
    K.set_session(sess)
    with tf.name_scope('writer'):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(save_log_path, sess.graph)
    
    with tf.name_scope('intialization'):
        sess.run(init_var)
        sess.run(init_data)
        sess.run(init_op, init_feed_dict)

    if (FLAGS.finetune):
        feature_saver.restore(sess, FLAGS.encoder_model)
        decoder_saver.restore(sess, FLAGS.decoder_model)


    start_time = time.time()

    if FLAGS.use_external == True:
        testing_interval = 1000
    else:
        testing_interval = 200

    if FLAGS.mode == "train":
        # ipdb.set_trace()
        for step in range(FLAGS.max_steps):
            # Perform testing
            if (step) % testing_interval == 0:
                print('Testing.....')

                test_images = []
                test_predictions = []
                test_ids = []
                soft_labels = []
                sess.run(init_test)

                test_predictions_ag = []
                soft_labels_ag =[]

                rr = 0
                rn = 0
                nr = 0
                nn = 0

                # Do multi-scale aggregation
                scales = [1.0, 1.5, 2.0]

                def multi_scale_aggregation(input_images):
                    batch_size, h, w, _ = input_images.shape
                    valid_soft_lst = []
                    valid_predict_lst = []
                    for i in range(batch_size):
                        image = input_images[i, :, :, :]

                        soft_labels = []
                        for scale in scales:
                            h_new = int(h * scale)
                            w_new = int(w * scale)
                            image_scaled = cv2.resize(image, (h_new, w_new), interpolation=cv2.INTER_LINEAR)
                            image_scaled = Image.fromarray(image_scaled)
                            for idx in range(1):
                                image_rot = np.array(image_scaled.rotate(idx * 90))[None, ...]
                                predict, soft = sess.run([pred_hard, soft_label],
                                                         feed_dict={input_placeholder: image_rot})
                                # predict = np.array((Image.fromarray(predict[0, :, :, :])).rotate(-idx*90))
                                # soft = np.array((Image.fromarray(soft[0, :, :, :])).rotate(-idx*90))
                                predict = ndimage.rotate(predict[0, :, :, :], -idx * 90)
                                soft = ndimage.rotate(soft[0, :, :, :], -idx * 90)

                                soft_labels.append(cv2.resize(soft, (h, w), interpolation=cv2.INTER_NEAREST)[..., None])

                        # ipdb.set_trace()
                        # Average the predictions
                        soft_label_final = np.concatenate(tuple(soft_labels), axis=-1)
                        soft_label_final = np.mean(soft_label_final, axis=-1)
                        valid_soft_lst.append(soft_label_final[None, ...])
                        valid_predict_lst.append(np.argmax(soft_label_final, axis=-1)[None, ...])

                    # Concat back to batch
                    valid_soft = np.concatenate(tuple(valid_soft_lst), axis=0)
                    valid_predict = np.concatenate(tuple(valid_predict_lst), axis=0)[..., None]

                    return valid_soft, valid_predict

                try:
                    while True:
                        ###################
                        ## Original part ##
                        ###################
                        if (FLAGS.use_external):
                            test_image, test_labels, ids = sess.run(next_test)
                        else:
                            test_image, ids = sess.run(next_test)

                        test_predict, test_soft = sess.run([pred_hard, soft_label], feed_dict={input_placeholder : test_image})
                        test_predictions.extend(test_predict)

                        if (FLAGS.use_external):
                            rr += np.sum((test_predict == 1) * (test_labels == 1))
                            rn += np.sum((test_predict == 1) * (test_labels == 0))
                            nr += np.sum((test_predict == 0) * (test_labels == 1))
                            nn += np.sum((test_predict == 0) * (test_labels == 0))

                        test_images.extend(test_image)
                        test_ids.extend(ids)
                        soft_labels.extend(test_soft)
                        print(ids)

                        ######################
                        ## Aggregation part ##
                        ######################
                        test_soft2, test_predict2 = multi_scale_aggregation(test_image)
                        soft_labels_ag.extend(test_soft2)

                except tf.errors.OutOfRangeError:
                    print('[INFO] Test Done.')


                if (FLAGS.save_csv_file):
                    print('[INFO] Write to csv file')
                    create_submission_files(test_predictions, test_ids, save_output_path, step)
                if (FLAGS.save_soft_masks):
                    print('[INFO] Generating soft label masks')
                    create_soft_label_mask(soft_labels, save_output_path, step)
                    create_soft_label_mask_ag(soft_labels_ag, save_output_path, step)

                # if (FLAGS.use_external):
                #     mIoU = ( 0.5 * rr / (rr + rn + nr) ) + ( 0.5 * nn / (nn + rn + nr) )
                #     print('[INFO] Validation mIoU')
                #     print (mIoU)
                # else :
                # Perform validation
                sess.run(init_valid)
                valid_soft_labels = []
                accuracy_lst = []

                valid_soft_labels_ag = []
                accuracy_lst_ag = []

                rr_ag = 0
                rn_ag = 0
                nr_ag = 0
                nn_ag = 0

                try:
                    while True:
                        # Compute mIoU
                        valid_image, valid_labels = sess.run(next_valid)
                        valid_predict, valid_soft = sess.run([pred_hard, soft_label], feed_dict={input_placeholder : valid_image})
                        rr += np.sum((valid_predict == 1) * (valid_labels == 1))
                        rn += np.sum((valid_predict == 1) * (valid_labels == 0))
                        nr += np.sum((valid_predict == 0) * (valid_labels == 1))
                        nn += np.sum((valid_predict == 0) * (valid_labels == 0))
                        valid_soft_labels.extend(valid_soft)

                        # Compute accuracy
                        valid_accuracy = np.mean((valid_predict == valid_labels).astype(np.int32))
                        accuracy_lst.append(valid_accuracy)

                        valid_soft, valid_predict = multi_scale_aggregation(valid_image)
                        # ipdb.set_trace()
                        rr_ag += np.sum((valid_predict == 1) * (valid_labels == 1))
                        rn_ag += np.sum((valid_predict == 1) * (valid_labels == 0))
                        nr_ag += np.sum((valid_predict == 0) * (valid_labels == 1))
                        nn_ag += np.sum((valid_predict == 0) * (valid_labels == 0))

                        valid_accuracy = np.mean((valid_predict == valid_labels).astype(np.int32))

                        valid_soft_labels_ag.append(valid_soft)
                        accuracy_lst_ag.append(valid_accuracy)

                except tf.errors.OutOfRangeError:
                    print('[INFO] Valid Done.')

                # Original
                mIoU = (0.5 * rr / (rr + rn + nr)) + (0.5 * nn / (nn + rn + nr))
                valid_accuracy = np.mean(np.array(accuracy_lst))

                # Aggregated
                mIoU_ag = (0.5 * rr_ag / (rr_ag + rn_ag + nr_ag)) + (0.5 * nn_ag / (nn_ag + rn_ag + nr_ag))
                valid_accuracy_ag = np.mean(np.array(accuracy_lst_ag))

                # ipdb.set_trace()

                print('[INFO] Validation Reslults')
                print("\t mIoU:", mIoU)
                print("\t Accuracy:", valid_accuracy)
                if (FLAGS.save_soft_masks):
                    create_valid_soft_label(valid_soft_labels, save_output_path, step)

                # ipdb.set_trace()
                # Record summary
                val_summary = tf.Summary()
                val_summary.value.add(tag="val_accuracy", simple_value=valid_accuracy)
                val_summary.value.add(tag="val_mIoU", simple_value=mIoU)
                val_summary.value.add(tag="val_accuracy_ag", simple_value=valid_accuracy_ag)
                val_summary.value.add(tag="val_mIoU_ag", simple_value=mIoU_ag)

                writer.add_summary(val_summary, global_step=step)

            #Training loop
            images, labels = sess.run(next_data)

            if (step) % 50 == 0 :
                summary, _, _ = sess.run([merged, feature_opt, decoder_opt],
                                feed_dict={input_placeholder: images, label_placeholder: labels,
                                           test_rgb_placeholder: test_images[0:10],
                                           test_mask_placeholder: test_predictions[0:10]})
                duration = time.time() - start_time
                print('Step %d: %.3f sec' % (step, duration))

                writer.add_summary(summary, step)
                start_time = time.time()

            else:
                _, _ = sess.run([feature_opt, decoder_opt],
                                feed_dict={input_placeholder: images, label_placeholder: labels,
                                           test_rgb_placeholder: test_images[0:10],
                                           test_mask_placeholder: test_predictions[0:10]})


            if (step) % 1000 == 0 and not step == 0 :
                feature_saver.save(sess, os.path.join(save_model_path, 'encoder.ckpt'), global_step = step)
                decoder_saver.save(sess, os.path.join(save_model_path, 'decoder.ckpt'), global_step = step)

    # Pure testing mode
    elif FLAGS.mode == "test":
        print('Testing.....')

        test_images = []
        test_predictions = []
        test_ids = []
        soft_labels = []
        sess.run(init_test)

        test_predictions_ag = []
        soft_labels_ag = []

        rr = 0
        rn = 0
        nr = 0
        nn = 0

        # Do multi-scale aggregation
        scales = [1.0, 1.5, 2.0]

        def multi_scale_aggregation(input_images):
            batch_size, h, w, _ = input_images.shape
            valid_soft_lst = []
            valid_predict_lst = []
            for i in range(batch_size):
                image = input_images[i, :, :, :]

                soft_labels = []
                for scale in scales:
                    h_new = int(h * scale)
                    w_new = int(w * scale)
                    image_scaled = cv2.resize(image, (h_new, w_new), interpolation=cv2.INTER_LINEAR)
                    image_scaled = Image.fromarray(image_scaled)
                    for idx in range(1):
                        image_rot = np.array(image_scaled.rotate(idx * 90))[None, ...]
                        predict, soft = sess.run([pred_hard, soft_label],
                                                 feed_dict={input_placeholder: image_rot})
                        # predict = np.array((Image.fromarray(predict[0, :, :, :])).rotate(-idx*90))
                        # soft = np.array((Image.fromarray(soft[0, :, :, :])).rotate(-idx*90))
                        predict = ndimage.rotate(predict[0, :, :, :], -idx * 90)
                        soft = ndimage.rotate(soft[0, :, :, :], -idx * 90)

                        soft_labels.append(cv2.resize(soft, (h, w), interpolation=cv2.INTER_NEAREST)[..., None])

                # ipdb.set_trace()
                # Average the predictions
                soft_label_final = np.concatenate(tuple(soft_labels), axis=-1)
                soft_label_final = np.mean(soft_label_final, axis=-1)
                valid_soft_lst.append(soft_label_final[None, ...])
                valid_predict_lst.append(np.argmax(soft_label_final, axis=-1)[None, ...])

            # Concat back to batch
            valid_soft = np.concatenate(tuple(valid_soft_lst), axis=0)
            valid_predict = np.concatenate(tuple(valid_predict_lst), axis=0)[..., None]

            return valid_soft, valid_predict

        try:
            while True:
                ###################
                ## Original part ##
                ###################
                if (FLAGS.use_external):
                    test_image, test_labels, ids = sess.run(next_test)
                else:
                    test_image, ids = sess.run(next_test)

                test_predict, test_soft = sess.run([pred_hard, soft_label], feed_dict={input_placeholder: test_image})
                test_predictions.extend(test_predict)

                if (FLAGS.use_external):
                    rr += np.sum((test_predict == 1) * (test_labels == 1))
                    rn += np.sum((test_predict == 1) * (test_labels == 0))
                    nr += np.sum((test_predict == 0) * (test_labels == 1))
                    nn += np.sum((test_predict == 0) * (test_labels == 0))

                test_images.extend(test_image)
                test_ids.extend(ids)
                soft_labels.extend(test_soft)
                print(ids)

                ######################
                ## Aggregation part ##
                ######################
                test_soft2, test_predict2 = multi_scale_aggregation(test_image)
                soft_labels_ag.extend(test_soft2)

        except tf.errors.OutOfRangeError:
            print('[INFO] Test Done.')

        if (FLAGS.save_csv_file):
            print('[INFO] Write to csv file')
            create_submission_files(test_predictions, test_ids, save_output_path, 0)
        if (FLAGS.save_soft_masks):
            print('[INFO] Generating soft label masks')
            create_soft_label_mask(soft_labels, save_output_path, 0)
            create_soft_label_mask_ag(soft_labels_ag, save_output_path, 0)

        # if (FLAGS.use_external):
        #     mIoU = ( 0.5 * rr / (rr + rn + nr) ) + ( 0.5 * nn / (nn + rn + nr) )
        #     print('[INFO] Validation mIoU')
        #     print (mIoU)
        # else :
        # Perform validation
        sess.run(init_valid)
        valid_soft_labels = []
        accuracy_lst = []

        valid_soft_labels_ag = []
        accuracy_lst_ag = []

        rr_ag = 0
        rn_ag = 0
        nr_ag = 0
        nn_ag = 0

        try:
            while True:
                # Compute mIoU
                valid_image, valid_labels = sess.run(next_valid)
                valid_predict, valid_soft = sess.run([pred_hard, soft_label],
                                                     feed_dict={input_placeholder: valid_image})
                rr += np.sum((valid_predict == 1) * (valid_labels == 1))
                rn += np.sum((valid_predict == 1) * (valid_labels == 0))
                nr += np.sum((valid_predict == 0) * (valid_labels == 1))
                nn += np.sum((valid_predict == 0) * (valid_labels == 0))
                valid_soft_labels.extend(valid_soft)

                # Compute accuracy
                valid_accuracy = np.mean((valid_predict == valid_labels).astype(np.int32))
                accuracy_lst.append(valid_accuracy)

                valid_soft, valid_predict = multi_scale_aggregation(valid_image)
                # ipdb.set_trace()
                rr_ag += np.sum((valid_predict == 1) * (valid_labels == 1))
                rn_ag += np.sum((valid_predict == 1) * (valid_labels == 0))
                nr_ag += np.sum((valid_predict == 0) * (valid_labels == 1))
                nn_ag += np.sum((valid_predict == 0) * (valid_labels == 0))

                valid_accuracy = np.mean((valid_predict == valid_labels).astype(np.int32))

                valid_soft_labels_ag.append(valid_soft)
                accuracy_lst_ag.append(valid_accuracy)

        except tf.errors.OutOfRangeError:
            print('[INFO] Valid Done.')

        # Original
        mIoU = (0.5 * rr / (rr + rn + nr)) + (0.5 * nn / (nn + rn + nr))
        valid_accuracy = np.mean(np.array(accuracy_lst))

        # Aggregated
        mIoU_ag = (0.5 * rr_ag / (rr_ag + rn_ag + nr_ag)) + (0.5 * nn_ag / (nn_ag + rn_ag + nr_ag))
        valid_accuracy_ag = np.mean(np.array(accuracy_lst_ag))

        print('[INFO] Validation Reslults')
        print("\t mIoU:", mIoU)
        print("\t Accuracy:", valid_accuracy)
        print('[INFO] Multi-scale Aggregation Results')
        print("\t mIoU", mIoU_ag)
        print("\t Accuracy:", valid_accuracy_ag)
        if (FLAGS.save_soft_masks):
            create_valid_soft_label(valid_soft_labels, save_output_path, 0)

    else:
        raise ValueError("Unknow mode! Use 'train' or 'test' instead...")


if __name__ == '__main__':
    if not tf.gfile.Exists(FLAGS.save_path):
        tf.gfile.MakeDirs(FLAGS.save_path)
    previous_runs = os.listdir(FLAGS.save_path)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = len(previous_runs) + 1
        
    # rundir = 'run_%02d' % run_number
    rundir = FLAGS.experiment_name
    save_log_path = os.path.join(FLAGS.save_path, rundir + '/logs')
    save_model_path = os.path.join(FLAGS.save_path, rundir + '/model')
    save_output_path = os.path.join(FLAGS.save_path, rundir + '/output')

    tf.gfile.MakeDirs(save_log_path)
    tf.gfile.MakeDirs(save_model_path)
    tf.gfile.MakeDirs(save_output_path)

    main()
