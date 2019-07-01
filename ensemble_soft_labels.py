from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pickle
import numpy as np
import os
import csv
import ipdb

def parse_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


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


if __name__ == "__main__":
    # newsampling2_class_final
    soft_label1 = "./experiments/deeplabv3+_lowlr_newsampling2_class_final/output/softLabels_ag_2600.pkl"
    # newsampling2 focal final
    soft_label2 = "./experiments/deeplabv3+_lowlr_newsampling2_focal_final/output/softLabels_ag_2600.pkl"
    # newsampling class final
    soft_label3 = "./experiments/deeplabv3+_lowlr_newsampling_class_final/output/softLabels_ag_2400.pkl"
    # newsampling2 wclass tv
    soft_label4 = "./experiments/deeplabv3+_lowlr_newsampling2_wclass_tv_final/output/softLabels_ag_2400.pkl"

    soft_label5 = "./experiments/deeplabv3+_lowlr_newsampling2_class/output/softLabels_3200.pkl"

    soft_label6 = "./experiments/deeplabv3+_lowlr_newsampling2_wclass_tv_final_debug/output/softLabels_ag_1400.pkl"

    path_lst = [
        soft_label1,
        soft_label2,
        soft_label3,
        soft_label4,
        soft_label5,
        soft_label6
    ]

    # Parse all the labels
    soft_label_lst = [np.array(parse_pkl(_))[None, ...] for _ in path_lst]

    # Concat along 1-st dimensions
    soft_label = np.concatenate(tuple(soft_label_lst), axis=0)
    soft_label = np.mean(soft_label, axis=0)

    # Get the predictions
    predictions = np.argmax(soft_label, axis=-1)

    # test ids
    test_ids = [10, 105, 106, 107, 108, 11] + \
               [115, 116,  12, 121, 122, 123] + \
               [124, 128, 129, 130, 131, 136] + \
               [137, 138, 139,  14, 140, 142] + \
               [143, 144, 145,  15, 151, 152] + \
               [153, 154, 155, 157, 159, 161] + \
               [162, 168, 169, 170, 174, 175] + \
               [176, 177, 186, 187, 189, 190] + \
               [191, 192, 196, 200, 201, 202] + \
               [204, 205, 206, 207, 208,  21] + \
               [211, 215, 216, 218, 219, 220] + \
               [221, 222, 223,  23,  25, 26] + \
               [27, 29, 36, 40, 41, 49] + \
               [50, 51, 54, 61, 64, 65] + \
               [69,  7, 76, 79,  8, 80] + \
               [9, 90, 92, 93]
    ipdb.set_trace()
    create_submission_files(predictions, test_ids, output_path="./", count=5)
