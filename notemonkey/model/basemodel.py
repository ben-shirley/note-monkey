import keras
import numpy as np
import cv2

from components.word import Word
class BaseModel():
    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    def predict(self, word: Word) -> str:
        return None
    def _preprocess():  
        raise NotImplementedError

    def _postprocess():
        raise NotImplementedError