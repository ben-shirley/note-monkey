"""Main class for the project, combines all steps of the reading process"""
from line_segmenter.linesegmenter import LineSegmenter
from word_segmenter.base_word_segmenter import BaseWordSegmenter
from model.basemodel import BaseModel


class Reader():

    def __init__(self, line_segmenter: LineSegmenter,
                word_segmenter: BaseWordSegmenter,
                model: BaseModel):
        self.line_segmenter = line_segmenter
        self.word_segmenter = word_segmenter
        self.model = model

    def read(self, img):
        
        self.line_segmenter.segment(img)
        