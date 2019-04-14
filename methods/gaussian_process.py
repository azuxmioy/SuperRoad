from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
sys.path.append("../")

from util import *
import cv2
from methods.method import Method
from dataset import Dataset
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import ipdb


# Define Gaussian Process method
class gaussian_process(Method):
    def __init__(self, mode="cross_validation"):
        """
        :param mode: use "cross_validation" or normal "validation"
        """
        super(gaussian_process, self).__init__()
        self.model = None
        self.mode = mode
        assert self.mode in ["cross_validation", "validation"]

        # Define some parameters here
        self.keypoint_config = {
            "image_width": 400,
            "image_height": 400,
            "step_size": 5,
            "radius": 10
        }

    def train(self, train_input, train_output):
        # Convert data type and perform certain thresholding
        converted_data = self.convert_data(train_input, train_output)
        inputs = converted_data["inputs"]
        targets = converted_data["targets"]

        # Create training sets
        training_set = self.create_training_set(inputs, targets)
        ipdb.set_trace()

        raise NotImplementedError

    def test(self, test_input):
        # Check if model is trained
        if self.model is None:
            raise ValueError("[Error]: shouldn't execute test before training")
        pass

    def submit(self, test_input):
        pass

    # Create training set given gray scale images and labels
    def create_training_set(self, inputs, targets):
        # extract dense sift on every image
        feature_data_lst = []
        sift_extractor = cv2.xfeatures2d.SIFT_create()
        sample_map = self.get_sample_map()
        for idx in tqdm(range(len(inputs)), "[Info]: Extracting features from training set"):
            # Extract dsift on single image
            feature_single_image = self.extract_dsift(inputs[idx], targets[idx], sift_extractor, sample_map)
            feature_data_lst.append(feature_single_image)

        # Convert to array format
        feature_data_lst = np.asarray(feature_data_lst)

        # Create training and validation split
        training_set = {}
        if self.mode == "cross_validation":
            # create K-fold data
            kfold = KFold(n_splits=10, shuffle=True)

            for idx, (train_index, val_index) in enumerate(kfold.split(feature_data_lst)):
                training_set["fold_%02d"%(idx)] = {
                    "train_set": np.reshape(feature_data_lst[train_index, :], newshape=-1),
                    "val_set": np.reshape(feature_data_lst[val_index, :], newshape=-1)
                }

        else:
            # Create single train / val split
            train_set, val_set = train_test_split(feature_data_lst, train_size=90, shuffle=True)

            training_set = {
                "train_set": np.reshape(train_set, newshape=-1),
                "val_set": np.reshape(val_set, newshape=-1)
            }

        return training_set

    # ToDo: whether we want to use multi-scale feature
    # ToDo: whether we need to normalize sift to 0~1
    # Extract dsift on defined location given one input image and target pair
    def extract_dsift(self, input_image, label, sift_extractor, sample_map):
        # Extract dense sift based on the given sample map
        _, dense_sift = sift_extractor.compute(input_image, sample_map["cv"])

        # Get ground truth label
        label_lst = [label[loc[1], loc[0]] for loc in sample_map["tuple"]]

        feature_lst = []
        for idx in range(len(label_lst)):
            feature_lst.append({
                "feature": dense_sift[idx, :],
                "label": label_lst[idx]
            })

        return feature_lst


    # ToDo check if x, y in cv2 is consistent with numpy
    # Construct keypoint map
    def get_sample_map(self):
        # create sample point by the given step_size
        start_point = self.keypoint_config["step_size"]
        sample_map_cv2 = [cv2.KeyPoint(x, y, self.keypoint_config["radius"]) \
                for y in range(start_point, self.keypoint_config["image_height"], self.keypoint_config["step_size"]) \
                for x in range(start_point, self.keypoint_config["image_width"], self.keypoint_config["step_size"])]
        sample_map = [(x, y) \
                for y in range(start_point, self.keypoint_config["image_height"], self.keypoint_config["step_size"]) \
                for x in range(start_point, self.keypoint_config["image_width"], self.keypoint_config["step_size"])]

        return {
            "cv": sample_map_cv2,
            "tuple": sample_map
        }

    # Convert image to BGR and then gray scale. Labels to int 0 and 1 (mask)
    def convert_data(self, inputs, targets):
        # Convert to bgr and then to gray
        converted_inputs = [cv2.cvtColor(self.rgb_to_bgr(_), cv2.COLOR_BGR2GRAY) for _ in inputs]
        converted_targets = [(_.astype(np.float32) / 255. >= 0.5).astype(np.int16) for _ in targets]

        return {
            "inputs": converted_inputs,
            "targets": converted_targets
        }

    # Convert RGB array to BGR array
    @staticmethod
    def rgb_to_bgr(input_data):
        r_temp = input_data[:, :, 0, None]
        g_temp = input_data[:, :, 1, None]
        b_temp = input_data[:, :, 2, None]

        return np.concatenate((b_temp, g_temp, r_temp), axis=-1)


if __name__ == "__main__":
    # Load data
    dataset = Dataset("./dataset/train_input/", "./dataset/train_output/", "./dataset/test_input/")

    # Initialize method
    model = gaussian_process(mode="validation")
    model.train(dataset.train_input, dataset.train_output)

    # img = model.rgb_to_bgr(dataset.train_input[0])
    # step_size = 20
    # kp = [cv2.KeyPoint(x, y, 10) for y in range(0, 400, step_size)
    #       for x in range(0, 300, step_size)]
    #
    # img = cv2.drawKeypoints(cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY), kp, img)
    #
    # plt.imshow(img)
    # plt.show()




