from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import os
import re


# Training dataloader for cil dataset
def train_dataset(data_path, batch_size=16, shuffle=True, crop_size=[None, None], augment=False, buffer=30):
    def _load_image_label(image_path, label_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        label = tf.io.read_file(label_path)
        label = tf.io.decode_png(label)
        label = label / 255
        label = tf.math.round(label)

        # label = tf.div(tf.io.decode_png(label), 255 )
        label = tf.cast(label,tf.uint8)

        return image, label

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

    all_trian_image = os.listdir(data_path + 'train_input/')
    all_train_label = os.listdir(data_path + 'train_output/')

    all_trian_image_paths = sorted([ data_path + 'train_input/' + str(path) for path in all_trian_image])
    all_trian_label_paths = sorted([ data_path + 'train_output/' + str(path) for path in all_train_label])

    dataset = tf.data.Dataset.from_tensor_slices((all_trian_image_paths, all_trian_label_paths)).repeat()

    dataset = dataset.map(_load_image_label).prefetch(buffer)

    if shuffle:
        dataset = dataset.shuffle(1000)

    if crop_size[0] is not None:
        dataset = dataset.map(lambda image, label:
                        tuple( tf.py_function( _random_crop, [image, label, crop_size],
                        [tf.uint8, tf.uint8] )))

    if augment:
        dataset = dataset.map(lambda image, label:
                    tuple( tf.py_function( _flip, [image, label], [tf.uint8, tf.uint8] )))
        dataset = dataset.map(lambda image, label:
                    tuple( tf.py_function( _rotate, [image, label], [tf.uint8, tf.uint8] )))

    dataset = dataset.batch(batch_size)

    return dataset


# Testing dataloader for cil dataset
def test_dataset(data_path, batch_size=16, buffer=30):
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


# validation dataloader for cil dataset
def val_dataset(data_path, batch_size=16, buffer=30):
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

    all_valid_image = os.listdir(data_path + 'valid_input/')
    all_valid_label = os.listdir(data_path + 'valid_output/')

    all_valid_image_paths = sorted([ data_path + 'valid_input/' + str(path) for path in all_valid_image])
    all_valid_label_paths = sorted([ data_path + 'valid_output/' + str(path) for path in all_valid_label])

    dataset = tf.data.Dataset.from_tensor_slices((all_valid_image_paths, all_valid_label_paths))

    dataset = dataset.map(_load_image_label).prefetch(buffer)

    dataset = dataset.batch(batch_size)

    return dataset