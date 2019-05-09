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


gpunumber = 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(gpunumber)

flags = tf.app.flags

# Loading path
flags.DEFINE_boolean('use_external', True, 'using external dataset')
flags.DEFINE_string('data_path', '/mnt/sda/ganpizza/Spacenet_data/', 'directory of dataset')
flags.DEFINE_string('feature', 'xception/', 'directory feature extractor')
flags.DEFINE_string('tf_initial_checkpoint', 'xception/model.ckpt',
                    'The initial checkpoint in tensorflow format.')
                    
#flags.DEFINE_boolean('use_external', False, 'using external dataset')
#flags.DEFINE_string('data_path', 'dataset/', 'directory of dataset')
#flags.DEFINE_string('feature', 'resnet_v1_50/', 'directory feature extractor')

flags.DEFINE_boolean('finetune', False, 'using external dataset')
flags.DEFINE_string('encoder_model', 'pretrained/encoder.ckpt-10000', 'directory of encoder model')
flags.DEFINE_string('decoder_model', 'pretrained/decoder.ckpt-10000', 'directory of decoder model')

# Saving directory
flags.DEFINE_string('save_path', '/mnt/sda/ganpizza/runs', 'saving directotry')
flags.DEFINE_boolean('save_soft_masks', False, 'Saving test results')
flags.DEFINE_boolean('save_csv_file', False, 'Saving submission file')

# Training parameters
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('max_steps', 80000, 'Number of steps to run training.')
flags.DEFINE_float('f_lr', 1e-6, 'learning rate')
flags.DEFINE_float('d_lr', 1e-6, 'learning rate')
flags.DEFINE_float('alpha', 0.9, 'balance param')

# Deeplab param

flags.DEFINE_multi_integer('train_crop_size', None,
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


FLAGS = flags.FLAGS

save_log_path = None
save_model_path = None
save_output_path = None


def buildExternalDataset ( data_path, batch_size = 16, shuffle = True, crop_size = [512, 512], augment = False, buffer = 30):
    
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

    def _load_train_image(image_path, label_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        
        label = tf.io.read_file(label_path)
        label = tf.io.decode_jpeg(label)
        label = label / 150
        label = tf.math.round(label)
        label = tf.cast(label,tf.uint8)

        return image, label
    
    def _load_test_image(image_path, ids, size):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image)
        image = tf.random_crop(image, [size[0], size[1], tf.shape(image)[-1] ])
        return image, ids

    all_trian_image = os.listdir(data_path + 'RGB/')
    all_train_label = os.listdir(data_path + 'mask/')

    all_trian_image_paths = sorted([ data_path + 'RGB/' + str(path) for path in all_trian_image])
    all_trian_label_paths = sorted([ data_path + 'mask/' + str(path) for path in all_train_label])

    all_test_image_paths = all_trian_image_paths [-20:]
    all_test_label_paths = all_trian_label_paths [-20:]
    ids = [i for i in range(20)]


    all_trian_image_paths = all_trian_image_paths [:-20]
    all_trian_label_paths = all_trian_label_paths [:-20]

    train_dataset = tf.data.Dataset.from_tensor_slices((all_trian_image_paths, all_trian_label_paths)).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((all_test_image_paths, ids))

    train_dataset = train_dataset.map(_load_train_image).prefetch(buffer)

    if shuffle:
        train_dataset = train_dataset.shuffle(1000)

    if crop_size[0] is not None:
        train_dataset = train_dataset.map(lambda image, label: 
                        tuple( tf.py_function( _random_crop, [image, label, crop_size],
                        [tf.uint8, tf.uint8] )))

    if augment:
        train_dataset = train_dataset.map(lambda image, label:
                    tuple( tf.py_function( _flip, [image, label], [tf.uint8, tf.uint8] )))
        train_dataset = train_dataset.map(lambda image, label:
                    tuple( tf.py_function( _rotate, [image, label], [tf.uint8, tf.uint8] )))

    train_dataset = train_dataset.batch(batch_size)

    test_dataset = test_dataset.map(lambda image, ids: 
                        tuple( tf.py_function( _load_test_image, [image, ids, crop_size],
                        [tf.uint8, tf.int32] ))).prefetch(buffer)

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


def buildTrainDataset( data_path, batch_size = 16, shuffle = True, crop_size = [None, None], augment = False, buffer = 30):

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


def create_submission_files(test_predictions, test_ids, output_path, count):

    def patch_to_label(patch):
        # percentage of pixels > 1 required to assign a foreground label to a patch
        foreground_threshold = 0.25
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

    print(len(soft_masks))
    print(soft_masks[0].shape)

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

def main(_):

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

        feature_1 = train_endpoint['feature_extractor/xception_65/entry_flow/block2/unit_1/xception_module/'
                                   'separable_conv2_pointwise']
        feature_2 = train_endpoint['feature_extractor/xception_65/entry_flow/block3/unit_1/xception_module/'
                                   'separable_conv2_pointwise']

        feature_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'feature_extractor')


    def Decode(feature_map_1, feature_map_2, feature, scope):
        
        with tf.variable_scope(scope) as decoder_vs:

            conv_map1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(feature_map_1)
 
            #up_map2 = tf.keras.layers.UpSampling2D(size=(2, 2))(feature_map_2)
            up_map2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature_map_2)

            concat_map1 = tf.concat([conv_map1, up_map2], axis=3)
        
            #up_map3 = tf.keras.layers.UpSampling2D(size=(4, 4))(feature)
            up_map3 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(feature)

            conv_map2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(concat_map1)
            
            concat_map2 = tf.concat([conv_map2, up_map3], axis=3)

            conv_map3 = tf.keras.layers.Conv2D(2, (1, 1), kernel_initializer='he_normal', 
                        activation='linear', padding='same', strides=(1, 1))(concat_map2)
        
            #prediction = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_map3)
            prediction = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(conv_map3)
        
        return prediction

        
    with tf.name_scope('decoder_fn'):
        pred_soft = Decode(feature_1, feature_2, feature_3, scope = 'decoder')
        pred_hard = tf.expand_dims(tf.argmax (pred_soft, axis=3), -1 )
        soft_label = tf.nn.softmax(pred_soft, axis=3)

        decoder_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'decoder')


    with tf.name_scope('loss'):

        gt_onehot = tf.one_hot(tf.squeeze(label_placeholder, axis=3), depth = 2, dtype = tf.float32)
        class_loss = tf.nn.softmax_cross_entropy_with_logits(logits= pred_soft, labels= gt_onehot)
        class_loss = tf.reduce_mean(class_loss)

        smooth = 1e-9
        intersection = tf.reduce_sum(pred_soft * gt_onehot)
        union = tf.reduce_sum(pred_soft) + tf.reduce_sum(gt_onehot)

        jaccard_index = (intersection + smooth) / ( union - intersection + smooth) 
        jaccard_loss =  - tf.log(jaccard_index) 


        total_loss = FLAGS.alpha * class_loss + ( 1.0 - FLAGS.alpha) * jaccard_loss
        tf.summary.scalar('class_loss', class_loss) 
        tf.summary.scalar('jaccard_loss', jaccard_loss) 
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

        init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
                        FLAGS.tf_initial_checkpoint,
                        variables_to_restore,
                        ignore_missing_vars=True)


    with tf.name_scope('optimizer'):
        feature_opt = tf.train.AdamOptimizer(FLAGS.f_lr).minimize(total_loss, var_list = feature_varlist)
        decoder_opt = tf.train.AdamOptimizer(FLAGS.d_lr).minimize(total_loss, var_list = decoder_varlist)

    init_var = tf.global_variables_initializer()

    with tf.name_scope('saver'):
        feature_saver = tf.train.Saver(feature_varlist, max_to_keep=10)
        decoder_saver = tf.train.Saver(decoder_varlist, max_to_keep=10)

    with tf.name_scope('dataset'):

        if (FLAGS.use_external):
            train_data, test_data = buildExternalDataset( FLAGS.data_path, batch_size = FLAGS.batch_size,
                                        shuffle = True, crop_size = [400, 400],
                                        augment = True)
        else:
            train_data = buildTrainDataset( FLAGS.data_path, batch_size = FLAGS.batch_size,
                                        shuffle = True, crop_size = [400, 400],
                                        augment = True)
            test_data =  buildTestDataset( FLAGS.data_path, batch_size = FLAGS.batch_size)


        with tf.name_scope('dataset_iterator'):
            it = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            next_data = it.get_next()
            init_data = it.make_initializer(train_data)
            
            test_iterator = tf.data.Iterator.from_structure(test_data.output_types, test_data.output_shapes)
            next_test = test_iterator.get_next()
            init_test = test_iterator.make_initializer(test_data)

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

    for step in range(FLAGS.max_steps):

        if (step) % 1000 == 0:
            print('Testing.....')

            test_images = []
            test_predictions = []
            test_ids = []
            soft_labels = []
            sess.run(init_test)
            try:
                while True:
                    test_image, ids = sess.run(next_test)
                    test_predict, test_soft = sess.run([pred_hard, soft_label], feed_dict={input_placeholder : test_image})
                    test_predictions.extend(test_predict)
                    test_images.extend(test_image)
                    test_ids.extend(ids)
                    soft_labels.extend(test_soft)
                    print(ids)

            except tf.errors.OutOfRangeError:
                print('[INFO] Test Done.')

            if (FLAGS.save_csv_file):
                print('[INFO] Write to csv file')
                create_submission_files(test_predictions, test_ids, save_output_path, step)
            if (FLAGS.save_soft_masks):
                print('[INFO] Generating soft label masks')
                create_soft_label_mask(soft_labels, save_output_path, step)

        #Training loop
        
        images, labels = sess.run(next_data)
        
        summary, _ , _ =  sess.run([merged, feature_opt, decoder_opt],
                       feed_dict={input_placeholder: images, label_placeholder: labels,
                                  test_rgb_placeholder: test_images[0:10],
                                  test_mask_placeholder: test_predictions[0:10]})

        if (step) % 50 == 0 :
            duration = time.time() - start_time
            print('Step %d: %.3f sec' % (step, duration))

            writer.add_summary(summary, step)
            start_time = time.time()
        

        if (step) % 2500 == 0 and not step == 0 :
            feature_saver.save(sess, os.path.join(save_model_path, 'encoder.ckpt'), global_step = step)
            decoder_saver.save(sess, os.path.join(save_model_path, 'decoder.ckpt'), global_step = step)



if __name__ == '__main__':
        
    if not tf.gfile.Exists(FLAGS.save_path):
        tf.gfile.MakeDirs(FLAGS.save_path)
    previous_runs = os.listdir(FLAGS.save_path)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = len(previous_runs) + 1
        
    rundir = 'run_%02d' % run_number
    save_log_path = os.path.join(FLAGS.save_path, rundir + '/logs')
    save_model_path = os.path.join(FLAGS.save_path, rundir + '/model')
    save_output_path = os.path.join(FLAGS.save_path, rundir + '/output')

    tf.gfile.MakeDirs(save_log_path)
    tf.gfile.MakeDirs(save_model_path)
    tf.gfile.MakeDirs(save_output_path)

    tf.app.run()
    