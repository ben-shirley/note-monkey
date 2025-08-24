import numpy as np
from model.basemodel import BaseModel

class BaseWordSegmenter():

    def __init__(self, verbosity:int = 0):
        self.verbosity = verbosity

    def segment(self, image: np.ndarray):
        raise NotImplementedError()