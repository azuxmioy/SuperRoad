from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
sys.path.append("../")

from util import *
from methods.method import Method
from dataset import Dataset
import ipdb


# Define Gaussian Process method
class gaussian_process(Method):
    def __init__(self):
        super(gaussian_process, self).__init__()

    def train(self, train_input, train_output):
        pass

    def test(self, test_input):
        pass

    def submit(self, test_input):
        pass


if __name__ == "__main__":
    # Load data

