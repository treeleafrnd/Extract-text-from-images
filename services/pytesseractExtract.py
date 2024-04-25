import cv2
from pytesseract import pytesseract, Output
import numpy as np


class PytesseractExtract:
    def __init__(self):
        self.config = '--oem 3 --psm 3'  # Custom Tesseract configuration.

    def extract_text(self, image):
        """Preprocesses the image, performs OCR using Tesseract, and returns the extracted text and bounding boxes."""
        gray_image = gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(gray_image, config=self.config)
        boxes = self.get_text_boxes(image)
        return extracted_text, boxes
    def get_text_boxes(self, image):
        """ Parses Tesseract output to get bounding boxes for detected text."""
        d = pytesseract.image_to_data(image, output_type=Output.DICT)
        boxes = []
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                boxes.append((x, y, x + w, y + h))
        return boxes
