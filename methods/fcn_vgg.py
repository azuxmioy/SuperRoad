from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


class fcn_vgg(object):
    def __init__(self, config, data, mode="train"):
        # Initialize data and configs
        self.config = config
        self.data = data
        self.mode = mode

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
        pass

    def eval_graph(self):
        pass

    def inference_graph(self):
        pass