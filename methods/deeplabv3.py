from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
from deeplab import common
from deeplab import model


class deeplabv3(object):
    def __init__(self, config, data, mode="train", model_mode="deeplabv3"):
        # Initialize data and configs
        if not model_mode in ["deeplabv3", "deeplabv3+"]:
            raise ValueError("[Error] model mode is not supported for deeplabv3")

        self.model_mode = model_mode
        self.config = config
        self.data = data
        self.mode = mode

        # Initialize model options
        self.model_options1 = common.ModelOptions(
            outputs_to_num_classes=2,
            crop_size=None,
            atrous_rates=config.atrous_rates,
            output_stride=config.output_stride)

        # Build train graph
        if self.mode == "train":
            self.train_outputs = self.train_graph()
            self.eval_outputs = self.eval_graph()

        elif self.mode == "val":
            self.eval_outputs = self.eval_graph()

        elif self.mode == "inference":
            self.inference_outputs = self.inference_graph()

        else:
            raise ValueError("[Error] Unknown model mode.")

    def train_graph(self):
        # Fetch the training inputs and labels
        with tf.name_scope("train_inputs"):
            train_inputs = self.data["train_images"]

        with tf.name_scope("train_labels"):
            train_labels = self.data["train_labels"]

        # Build the feature extractor
        # ToDo: implement different extractor architecture in [resnet, xception, etc. ]
        with tf.variable_scope("feature_extractor"):
            # Extract features from different layers
            feature_3, train_endpoint = model.extract_features(train_inputs, self.model_options1, is_training=True)
            feature_1 = train_endpoint['feature_extractor/resnet_v1_50/block1/unit_2/bottleneck_v1/conv3']
            feature_2 = train_endpoint['feature_extractor/resnet_v1_50/block2/unit_2/bottleneck_v1/conv3']

            # Get feature extractor variables
            feature_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_extractor')

        # Build the decoder
        with tf.name_scope('decoder_fn'):
            # Get the predictions
            pred_soft = self.decoder(feature_1, feature_2, feature_3, scope='decoder')
            pred_hard = tf.expand_dims(tf.argmax(pred_soft, axis=3), -1)
            soft_label = tf.nn.softmax(pred_soft, axis=3)

            # Get decoder variables
            decoder_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')

        # Define the training losses
        with tf.name_scope('loss'):
            raise NotImplementedError

    def eval_graph(self):
        pass

    def inference_graph(self):
        pass

    @staticmethod
    def decoder(feature_map1, feature_map2, feature, scope):
        with tf.variable_scope(scope):
            conv_map1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(feature_map1)
            conv_map1 = tf.keras.layers.Activation('relu')(conv_map1)

            up_map2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(feature_map2)

            concat_map1 = tf.concat([conv_map1, up_map2], axis=3)

            up_map3 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='nearest')(feature)

            conv_map2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(concat_map1)
            conv_map2 = tf.keras.layers.Activation('relu')(conv_map2)

            concat_map2 = tf.concat([conv_map2, up_map3], axis=3)

            conv_map3 = tf.keras.layers.Conv2D(2, (1, 1), kernel_initializer='he_normal',
                                               activation='linear', padding='same', strides=(1, 1))(concat_map2)

            prediction = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(conv_map3)

        return prediction