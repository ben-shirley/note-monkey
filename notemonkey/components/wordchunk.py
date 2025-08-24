"""class that contains the smallest unit of segmented code,
a connected piece of text."""
import numpy as np
import cv2

class Chunk():
    def __init__(self, image: np.ndarray):
        self.image = image
