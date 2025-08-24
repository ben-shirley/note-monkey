import numpy as np

from components.wordchunk import Chunk

class Word():

    def __init__(self, image: np.ndarray, chunks:list[Chunk]=None):
        self.image = image
        self.chunks = chunks
        self.value = None

    def set_chunks(self, chunks: list[Chunk]):
        self.chunks = chunks

    def get_chunks(self):
        if self.chunks is not None:
            return self.chunks
        else:
            raise RuntimeError("This word has no assigned chunks. It may need additional segmentation")

    def set_value(self, value:str):
       self.value = value 
    
    def get_value(self):
        if self.value is not None:
            return self.value
        else:
            raise RuntimeError("This word has no assigned value. It may need classifying")