"""
Class implementing a convolutional recurrent neural network 
text classifier.

The idea is that the cnn layers search over the text to extract key features.
The recurrent neural network layers then sort over those features to determine the most likely string of text.
Finally, we examine the rnn output to find the most likely output.
"""
import torch
torch.backends.cudnn.enabled = True 

from PIL import Image
import numpy as np
import cv2

from model.basemodel import BaseModel
from model.crnn import CRNN
from components.word import Word

NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01

class LSTMModel(BaseModel):
    
    def __init__(self, verbosity=0, modelpath="models/lstm_model_v1.pt"):
        super().__init__(verbosity)

        self.CHARS = 'abcdefghijklmnopqrstuvwxyz'+'abcdefghijklmnopqrstuvwxyz'.upper() + "0123456789 "
        self.CHAR2LABEL = {char: i + 1 for i, char in enumerate(self.CHARS)}
        self.LABEL2CHAR = {label: char for char, label in self.CHAR2LABEL.items()}

        self.expected_width = 100
        self.expected_height = 32

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRNN(1, self.expected_height, self.expected_width, len(self.CHARS)+1)
        self.model.load_state_dict(torch.load(modelpath, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self, 
        word: Word
    ) -> str:
        """
        Given a word, returns the most likely string it could be.
        """
        processed_image = self._preprocess(word.image)
        logits = self.model(processed_image.cuda())
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        preds = self.ctc_decode(log_probs)
        if self.verbosity >= 2:
            print(preds)

        return ''.join(preds[0])

    def _preprocess(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """We apply the same preprocessing steps used to train the model."""
        if self.verbosity >= 4:
            cv2.imshow("old image", image)
            cv2.waitKey(0)
        image = Image.fromarray(image)
        image = image.resize((self.expected_width, self.expected_height), resample=Image.BILINEAR)
        image = np.array(image)

        if self.verbosity >= 3:
            cv2.imshow("new image", image)
            cv2.waitKey(0)
        
        # Reshape to (1, 1, height, width)
        image = image.reshape(1, 1, self.expected_height, self.expected_width)
        image = torch.FloatTensor(image).to(self.device)
        return image

    def _reconstruct(
        self, 
        labels, 
        blank=0
    ) -> list[str]:
        """
        Examines model outputs. Returns the most likely string (by deleting repeated letters)
        """
        new_labels = []
        # merge same labels
        previous = None
        for l in labels:
            if l != previous:
                new_labels.append(l)
                previous = l
        # delete blank
        new_labels = [l for l in new_labels if l != blank]

        return new_labels

    def _greedy_decode(
        self, 
        emission_log_prob, 
        blank=0, 
    ) -> list[str]:
        labels = np.argmax(emission_log_prob, axis=-1)
        labels = self._reconstruct(labels, blank=blank)
        return labels

    def ctc_decode(
        self, 
        log_probs, 
        blank=0, 
        method='greedy', 
        beam_size=10
        ) -> list[str]:
        """decodes model emissions"""
        emission_log_probs = np.transpose(log_probs.cpu().detach().numpy(), (1, 0, 2))

        decoders = {
            'greedy': self._greedy_decode,
        }
        decoder = decoders[method]

        decoded_list = []
        for emission_log_prob in emission_log_probs:
            decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
            decoded = [self.LABEL2CHAR[l] for l in decoded]
            decoded_list.append(decoded)
        return decoded_list

