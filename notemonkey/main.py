import numpy as np
import cv2

from line_segmenter.processinglinesegmenter import ProcessingLineSegmenter
from word_segmenter.dpwordsegmenter import DPWordChunkSegmenter
from word_segmenter.connected_component_segmenter import ConnectedComponentWordSegmenter
from model.emnist_classifier import EMNISTModel
from model.basemodel import BaseModel
from model.lstm_model import LSTMModel
from components.line import Line
from imagehandler import ImageHandler, DeliveryMode
import preprocessor

from reader import Reader

if __name__ == "__main__":
    # model = EMNISTModel('models/model_v5.keras', 'models/class_mapping.txt', verbosity=0)
    model = LSTMModel(verbosity=0)
    word_segmenter = ConnectedComponentWordSegmenter(verbosity=0)
    line_segmenter = ProcessingLineSegmenter(verbosity=0)
    reader = Reader(line_segmenter, word_segmenter, model)

    handler = ImageHandler("datasets/misc-images/")
    handler.image_delivery_mode = DeliveryMode.IN_ORDER
    image = handler.get_new_image()
    image = handler.get_new_image()
    for i in range(6):
        image = handler.get_new_image()
        
        desired_width = 1500
        desired_height = int(image.shape[0] * (desired_width / image.shape[1]))
        image = cv2.resize(image, ( desired_width, desired_height))
        image = preprocessor.resize_img(image, resize_factor=1)

        lines = reader.read(image)

        output = ""
        for line in lines:
            # currently I am assuming that the text can be perfectly segmented, but
            # this does not happen all the time currently
            for word in line.words:
                output += word.get_value()
                output += " "
            output += "\n"
        print(output)
        handler.show_image(preprocessor.resize_img(image, resize_factor=0.25))
    