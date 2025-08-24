"""
This is my attempt at making a segementer that segments connected components using
opencv rather than dynamic programming which I used before. It will definitately oversegment,
but I am hoping to do some post processing to fix that.
"""
import numpy as np
import cv2

from word_segmenter.base_word_segmenter import BaseWordSegmenter
from model.basemodel import BaseModel
from components.wordchunk import Chunk
import preprocessor
from components.word import Word

class ConnectedComponentWordSegmenter(BaseWordSegmenter):
    
    def __init__(self, verbosity:int = 0):
        super().__init__(verbosity=verbosity)
    
    def segment(self, line_image: np.ndarray):
        """chops a line up into its connected components.
            Then, we analyse those components and their spacing to identify which
            components are likely to correspond to spaces."""

        if line_image.shape[-1] == 0:
            return []

        preprocessed_image = self._preprocess(line_image)
        
        segmented_sections = self._segment_into_connected_components(preprocessed_image)
        
        self.merge_high_overlap_sections(segmented_sections)

        word_segmented = self.split_into_words(segmented_sections)

        # finally, merge the chunks and return the words.
        words = []
        for word in word_segmented:
            last_chunk, last_pixel = word[-1]
            width = last_chunk.shape[1] + last_pixel
            height = last_chunk.shape[0]

            combined_image = np.zeros((height, width))
            for chunk, x_val in word:
                combined_image[:, x_val:x_val+chunk.shape[1]]=chunk
            combined_image = preprocessor.crop_image_tight(combined_image, direction="x", target_color=1)
            if self.verbosity >= 2:
                cv2.imshow("word", combined_image)
                cv2.waitKey(0)
            words.append(Word(combined_image, chunks=[Chunk(section[0]) for section in word])) 
            

        return words

    def _preprocess(self, image):
        binarized_img = preprocessor.preprocess_img(image) 
        inverted_img = 255 - binarized_img
        return inverted_img

        
    def _segment_into_connected_components(self, image: np.ndarray) -> list[tuple[np.ndarray, int]]:
        num_labels, labeled_img = cv2.connectedComponents(image)
        if self.verbosity >= 3:
            image_to_show = labeled_img.copy()
            label_hue = np.uint8(179*image_to_show/np.max(image_to_show))
            blank_ch = 255*np.ones_like(label_hue)
            image_to_show= cv2.merge([label_hue, blank_ch, blank_ch])

            image_to_show= cv2.cvtColor(image_to_show, cv2.COLOR_HSV2BGR)
            image_to_show[label_hue==0] = 0

            cv2.imshow('labeled.png', image_to_show)
            cv2.waitKey(0)
        

        if num_labels == 1 and image[0][0] != 0 :
            #TODO when there is only one connected component (ie a segmentation error), is now broken.
            raise NotImplementedError

        segmented_sections = []
        for i in range(1, num_labels):
            image_segment = labeled_img == i
            first_pixel = self._get_first_white_pixel_index(image_segment)
            image_segment = preprocessor.crop_image_tight(image_segment, direction='x', target_color=1) 

            segmented_sections.append((image_segment.astype(np.float32), first_pixel))

        segmented_sections = sorted(segmented_sections, key=lambda x: x[1]) 
        return segmented_sections


    def _get_first_white_pixel_index(self, image: np.ndarray, y_step:int=1):
        current_index = image.shape[1]
        for y_index in range(0, image.shape[0], y_step):
            for x_index in range(image.shape[1]):
                if image[y_index, x_index] == 1:
                    current_index = min(x_index, current_index)
                    break
        return current_index
    
    def merge_high_overlap_sections(self, sections, merge_modifier=0.1):
        """function that merges sections containing high overlap
        we do this by measuring the percentage that each image overlaps with its neighbours, and if it is high
        enough, then we merge them"""

        old_sections = None
        new_sections = sections
        iterations = 0
        while old_sections != new_sections or iterations < 100:
            iterations += 1
            old_sections = new_sections
            index=0
            while index < len(new_sections)-1:
                first_x_start = new_sections[index][1]
                next_x_start = new_sections[index+1][1]
                first_img_width = new_sections[index][0].shape[1]
                next_img_width = new_sections[index+1][0].shape[1]
                first_x_stop = first_x_start + first_img_width
                next_x_stop = next_x_start + next_img_width 

                overlap_start = max(first_x_start, next_x_start)
                overlap_end = min(first_x_stop, next_x_stop)
                overlap = max(0, overlap_end-overlap_start)
                
                if overlap/first_img_width >= merge_modifier or overlap/next_img_width >= merge_modifier:
                    # we should merge
                    new_img_start = min(new_sections[index][1], new_sections[index+1][1])
                    new_img_stop = max(new_sections[index][1]+new_sections[index][0].shape[1],\
                                  new_sections[index+1][1]+new_sections[index+1][0].shape[1])
                    new_img_width = new_img_stop - new_img_start
                    
                    new_img = np.zeros((new_sections[index][0].shape[0], new_img_width), np.uint8)
                    new_img[:, new_sections[index][1]-new_img_start:\
                            new_sections[index][1]-new_img_start+new_sections[index][0].shape[1]] |= (new_sections[index][0]).astype(np.uint8)
                    new_img[:, new_sections[index+1][1]-new_img_start:\
                            new_sections[index+1][1]-new_img_start+new_sections[index+1][0].shape[1]] |= (new_sections[index+1][0]).astype(np.uint8)

                    if self.verbosity >= 3:
                        print('merging high overlap')

                        cv2.imshow('old image', new_sections[index][0].astype(float))
                        cv2.imshow('old image 2', new_sections[index+1][0].astype(float))
                        cv2.imshow('merged image', new_img.astype(float))
                        cv2.waitKey(0)

                    new_sections.pop(index+1)
                    new_sections[index] = (new_img, new_img_start)
                else:
                    # if we did not merge two images, hence shortening our list, we need to move along by one
                    index += 1 

    def split_into_words(self, segmented_sections, segment_threshold=2) -> list[list[np.ndarray]]:
        """takes in the segmented sections, along with their starting x indexes, 
        and by considering the distances between their finishing points and the next character's starting point,
        decides whether there is likely to be a space""" 

        if len(segmented_sections) == 0:
            return []

        distances_between_characters = []
        for index in range(len(segmented_sections)-1):
            this_segment_finish_point = segmented_sections[index][1] + segmented_sections[index][0].shape[1]
            next_segment_start = segmented_sections[index+1][1]
            distances_between_characters.append(max(0, next_segment_start-this_segment_finish_point))
        
        distances_between_characters= np.array(distances_between_characters)
        # for now I'll take the median since it is easy to calculate and should be around about the standard segment gap
        # it might be worth considering other metrics however
        median_gap = np.median(distances_between_characters)
        
        words = []
        current_word = []
        for index in range(len(segmented_sections)-1):
            current_word.append(segmented_sections[index])
            if distances_between_characters[index] > segment_threshold*median_gap:
                # if there is a space
                words.append(current_word)
                current_word = []
        if len(current_word) == 0:
            words.append([segmented_sections[-1]])
        else:
            current_word.append(segmented_sections[-1])
            words.append(current_word)
        

        if self.verbosity >= 4:
            for word in words:
                for chunk, pixel in word:
                    cv2.imshow(f'{pixel}', chunk)
                    cv2.waitKey(0)

        return words

