from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import os


def external_dataset(data_path, batch_size=16, shuffle=True, crop_size=[None, None], resize=[512, 512],
                         augment=False, buffer=30):
    def _random_crop(image, label, size):

        combined = tf.concat([image, label], axis=2)
        last_label_dim = tf.shape(label)[-1]
        last_image_dim = tf.shape(image)[-1]
        combined_crop = tf.random_crop(combined,
                                       size=tf.concat([size, [last_label_dim + last_image_dim]], axis=0))
        return combined_crop[:, :, :last_image_dim], combined_crop[:, :, last_image_dim:]

    def _flip(image, label):
        combined = tf.concat([image, label], axis=2)
        last_label_dim = tf.shape(label)[-1]
        last_image_dim = tf.shape(image)[-1]

        combined_flip = tf.image.random_flip_left_right(combined, seed=0)
        combined_flip = tf.image.random_flip_up_down(combined_flip, seed=0)

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
        label = label / 150
        label = tf.math.round(label)
        label = tf.cast(label, tf.uint8)

        return image, label

    def _load_test_image(image_path, ids, size):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        image = tf.random_crop(image, [512, 512, tf.shape(image)[-1]])

        return image, ids

    all_trian_image = os.listdir(data_path + 'RGB/')
    all_train_label = os.listdir(data_path + 'real_mask/')

    all_trian_image_paths = sorted([data_path + 'RGB/' + str(path) for path in all_trian_image])
    all_trian_label_paths = sorted([data_path + 'real_mask/' + str(path) for path in all_train_label])

    all_test_image_paths = all_trian_image_paths[-50:]
    all_test_label_paths = all_trian_label_paths[-50:]
    ids = [i for i in range(50)]

    all_trian_image_paths = all_trian_image_paths[:-50]
    all_trian_label_paths = all_trian_label_paths[:-50]

    train_dataset = tf.data.Dataset.from_tensor_slices((all_trian_image_paths, all_trian_label_paths)).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((all_test_image_paths, ids))

    train_dataset = train_dataset.map(_load_train_image).prefetch(buffer)

    if shuffle:
        train_dataset = train_dataset.shuffle(1000)

    if crop_size[0] is not None:
        train_dataset = train_dataset.map(lambda image, label:
                                          tuple(tf.py_function(_random_crop, [image, label, crop_size],
                                                               [tf.uint8, tf.uint8])))

    if augment:
        train_dataset = train_dataset.map(lambda image, label:
                                          tuple(tf.py_function(_flip, [image, label], [tf.uint8, tf.uint8])))
        train_dataset = train_dataset.map(lambda image, label:
                                          tuple(tf.py_function(_rotate, [image, label], [tf.uint8, tf.uint8])))

    train_dataset = train_dataset.batch(batch_size)

    test_dataset = test_dataset.map(lambda image, label:
                                    tuple(tf.py_function(_load_test_image, [image, label, resize],
                                                         [tf.uint8, tf.int32]))).prefetch(buffer)

    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset