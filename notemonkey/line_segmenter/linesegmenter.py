from word_segmenter.base_word_segmenter import BaseWordSegmenter
import numpy as np
from components.line import Line

class LineSegmenter():
    
    def __init__(self, verbosity:int = 0):
        self.verbosity = verbosity

    def segment(image: np.ndarray) -> list[Line]:
        raise NotImplementedError()