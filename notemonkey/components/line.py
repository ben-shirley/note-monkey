import cv2
import numpy as np

import preprocessor
from word_segmenter.base_word_segmenter import BaseWordSegmenter
from components.word import Word


class Line():
    """class to represent an individual line
    of handwritten text
    
    NOTE: I am not sure if line should be doing all this processing, feels like that should be handled by the segmenter"""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.words=None
    
    def set_words(self, words: list[Word]):
        self.words = words
    
    def get_words(self):
        if self.words is not None:
            return self.words
        else:
            raise RuntimeError("this line has no set of associated words. It may need further segmenting and classification")

      