from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
from deeplab import common
from deeplab import model
from lib.loss import classification_loss, jaccard_loss


class deeplabv3(object):
    def __init__(self, config, data, mode="train", model_mode="deeplabv3"):
        # Initialize data and configs
        if not model_mode in ["deeplabv3"]:
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

        # Build train and validation graph
        if self.mode == "train":
            self.train_outputs = self.train_graph()
            self.eval_outputs = self.eval_graph()

        # Build the testing graph
        elif self.mode == "test":
            # Testing model must have pretrain weights
            assert self.config.pretrain is not None
            self.test_outputs = self.test_graph()

        else:
            raise ValueError("[Error] Unknown model mode.")

        # Initialize model from pre-trained weights
        if self.config.pretrain:



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

            # Get feature extractor variables
            feature_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_extractor')

        # Build the decoder
        with tf.name_scope('decoder_fn'):
            # Get the predictions
            pred_soft = self.decoder(feature_3, scope='decoder')
            pred_hard = tf.expand_dims(tf.argmax(pred_soft, axis=3), -1)
            soft_label = tf.nn.softmax(pred_soft, axis=3)

            # Get decoder variables
            decoder_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')

        # Define the training losses
        with tf.name_scope('loss'):
            # Use classification loss
            class_loss = classification_loss(logits=pred_soft, labels=train_labels)

            # Use jaccard loss
            jac_loss = jaccard_loss(logits=soft_label, labels=train_labels)

            # Weight the two lossess
            total_loss = self.config.alpha * class_loss + (1 - self.config.alpha) * jac_loss

        # Define the optimizer and train_op
        with tf.name_scope("train_op"):
            # Get or create global step
            global_step = tf.train.get_or_create_global_step()
            increase_global_step_op = tf.assign(global_step, global_step+1)

            # Defclare optimizers
            feature_optimizer = tf.train.AdamOptimizer(self.config.f_lr, self.config.momentum)
            decoder_optimizer = tf.train.AdamOptimizer(self.config.d_lr, self.config.momentum)

            # Compute gradients
            feature_grads = feature_optimizer.compute_gradients(total_loss, feature_varlist)
            decoder_grads = decoder_optimizer.compute_gradients(total_loss, decoder_varlist)

            # Update batchnorm and apply gradients
            updata_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updata_op):
                train_op_feature = feature_optimizer.apply_gradients(feature_grads)
                train_op_decoder = decoder_optimizer.apply_gradients(decoder_grads)

            train_op = tf.group([increase_global_step_op, train_op_feature, train_op_decoder])

        # Record the train summary
        with tf.name_scope("train_summary"):
            # Record scalar summary
            tf.summary.scalar("class_loss", class_loss, family="training_loss")
            tf.summary.scalar("jaccard_loss", jac_loss, family="training_loss")
            tf.summary.scalar("total_loss", total_loss, family="training_loss")

            # Record image summary
            tf.summary.image('input', tf.cast(train_inputs, tf.uint8), family="input_image")
            tf.summary.image('prediction', tf.cast(pred_hard, tf.float32), family="prediction_image")
            tf.summary.image('groundtruth', tf.cast(train_labels, tf.float32), family="groundtruth")

            # Merge the train summary
            train_summary = tf.summary.merge_all()

        return {
            "pred_soft": pred_soft,
            "pred_hard": pred_hard,
            "class_loss": class_loss,
            "jac_loss": jac_loss,
            "total_loss": total_loss,
            "train_op": train_op,
            "train_summary": train_summary
        }


    def eval_graph(self):
        # Fetch the validation inputs and labels
        with tf.name_scope("val_inputs"):
            val_inputs = self.data["val_images"]

        with tf.name_scope("val_labels"):
            val_labels = self.data["val_labels"]

        # Build the feature extractor
        # ToDo: implement different extractor architecture in [resnet, xception, etc. ]
        with tf.variable_scope("feature_extractor", reuse=tf.AUTO_REUSE):
            # Extract features from different layers
            feature_3, train_endpoint = model.extract_features(val_inputs, self.model_options1, is_training=True)

            # Get feature extractor variables
            feature_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_extractor')

        # Build the decoder
        with tf.name_scope('decoder_fn'):
            # Get the predictions
            pred_soft = self.decoder(feature_3, scope='decoder')
            pred_hard = tf.expand_dims(tf.argmax(pred_soft, axis=3), -1)
            soft_label = tf.nn.softmax(pred_soft, axis=3)

            # Get decoder variables
            decoder_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')

        return {
            "pred_soft": pred_soft,
            "pred_hard": pred_hard
        }

    def test_graph(self):
        raise NotImplementedError

    @staticmethod
    def decoder(feature, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            prediction = tf.keras.layers.Conv2D(2, (1, 1), kernel_initializer='he_normal',
                                                activation='linear', padding='same', strides=(1, 1))(feature)
            logits = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(prediction)

        return logits