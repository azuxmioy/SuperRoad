from util import *
from abc import ABC, abstractmethod

class Method(ABC):
    @abstractmethod
    def train(self, train_input, train_output):
        pass

    @abstractmethod
    def test(self, test_input):
        pass

    @abstractmethod
    def submit(self, test_input):
        pass

