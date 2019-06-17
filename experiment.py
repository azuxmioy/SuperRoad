from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

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
from tensorflow.contrib.slim.nets import resnet_utils
from deeplab.core import feature_extractor
from deeplab import common
from deeplab import model
from deeplab.utils import train_utils
from tensorflow.keras import backend as K

import argparse
import yaml
from collections import namedtuple
from dataset.cil_dataloader import train_dataset as cil_train_dataset
from dataset.cil_dataloader import test_dataset as cil_test_dataset
from dataset.external_dataloader import external_dataset
from methods.fcn_vgg import fcn_vgg
import ipdb


# Get the configuration file
def get_config(config_path):
    # Check if file exists
    if not os.path.exists(config_path):
        raise ValueError("[Error] The specified configuration file does not exists!")

    # Load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Print out the configuration
    print("[Info] Successfully load the configuration.")
    print("=============================")
    for key, value in config.items():
        print("\t", key, ": ", value)
    print("=============================")

    # Convert to namedtuple
    config = namedtuple("config", config.keys())(*config.values())

    return config


# Get the train, val, and test dataloader
def get_data(config=None):
    assert config is not None
    if config.dataset == "cil":
        # ToDo: we need cross validation in the training set
        # Declare the training and testing set
        train_dataset = cil_train_dataset(data_path=config.data_path,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          crop_size=config.train_crop_size,
                                          augment=True)
        test_dataset = cil_test_dataset(data_path=config.data_path,
                                        batch_size=1)

    elif config.dataset_name == "external":
        # ToDo: we need to check if the batch size parameter is only passed to train one
        train_dataset, test_dataset = external_dataset(data_path=config.data_path,
                                                       batch_size=config.batch_size,
                                                       shuffle=True,
                                                       crop_size=config.train_crop_size,
                                                       augment=True)

    else:
        raise ValueError("[Error] Unsupported dataset. Please check the loaded configurations.")

    # Create iterators
    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()

    # Get data
    # ToDo: we should have validation data
    train_images, train_labels = train_iterator.get_next()
    test_images = test_iterator.get_next()

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images
    }


# Get model by model name
# ToDo: only build inference graph for val and export
def get_model(config=None, data=None, mode="train"):
    assert config is not None
    assert data is not None

    # Check mode
    if mode == "train":
        if (not "train_images" in data.keys()) or (not "train_labels" in data.keys()):
            raise ValueError("[Error] train_images and train_labels are necessary for train mode!!")
    elif (mode == "val") or (mode == "test"):
        if not "test_images" in data.keys():
            raise ValueError("[Error] test_images is necessary for val and test mode!!")

    # Check models
    if config.model == "fcn_vgg":
        model = fcn_vgg(config, data, mode)

    elif config.model == "fcn_resnet":
        raise NotImplementedError
    elif config.model == "deeplab":
        raise NotImplementedError

    return model

# Training codes
def train(args=None, config=None):
    assert args is not None
    assert config is not None

    # Create output paths
    output_path = os.path.join(config.save_path, args.experiment_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, "logs"))
        os.makedirs(os.path.join(output_path, "model"))
        os.makedirs(os.path.join(output_path, "output"))

    # Create dataloader
    with tf.name_scope("data_loader"):
        data = get_data(config=config)

    # Create model
    with tf.name_scope("segmentation_model"):
        model = get_model(config=config, data=data, mode="train")


def main(args):
    # Load the configuration file
    config = get_config(args.config)

    if args.mode == "train":
        train(args=args, config=config)
    elif args.mode == "val":
        raise NotImplementedError
    elif args.mode == "export":
        raise NotImplementedError
    else:
        raise ValueError("[Error] Unknown mode. please choose between 'train', 'val', and 'export'")


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(description='DORN')
    parser.add_argument('--experiment_name', default=None, type=str,
                        help="The experiment name used to specify output path")
    parser.add_argument("--mode", default="train", type=str, help="mode: train, val, export")
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('--config', default="./config/batch3_new_sid.yaml", type=str,
                        help="The configuration file for all the training details")
    args = parser.parse_args()

    main(args)
