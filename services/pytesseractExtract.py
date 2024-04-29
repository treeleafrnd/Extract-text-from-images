import cv2
from pytesseract import pytesseract, Output
import numpy as np
import glob

class PytesseractExtract:
    def __init__(self):
        self.config = '--oem 3 --psm 3'  # Custom Tesseract configuration.


    def extract_text(self, image):
        gray_image = gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        #kernel = np.ones((5, 5), np.uint8)
        #denoised_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        image_height, image_width, _ = image.shape

        # Calculate aspect ratio
        aspect_ratio = image_width / image_height

        # Adjust configuration based on aspect ratio
        if aspect_ratio > 2:  # Wide image (potentially landscape invoices)
            self.config = '--oem 3 --psm 6'
        elif aspect_ratio < 1:  # Narrow image (potentially receipts)
            self.config = '--oem 1 --psm 8'



        text_regions = []
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_image)

        """Group connected components based on Y-coordinates (heuristic approach."""
        section_groups = {}
        section_threshold = 20  # Adjust threshold based on your invoice format
        for i in range(1, retval):
            x, y, w, h, area = stats[i]
            section_groups.setdefault(y // section_threshold, []).append((x, y, w, h))

        for section_key, components in section_groups.items():
            # Process each section group (potential invoice section)
            section_image =gray_image[components[0][1]:components[-1][1] + components[-1][3], :]  # Combine components vertically
            extracted_text, section_boxes = self._extract_text_and_boxes(section_image)
            text_regions.append((extracted_text, section_boxes))

        return text_regions

    def _extract_text_and_boxes(self, component):
        """Extracts text and bounding boxes directly from a single image component (potential text region)."""
        extracted_text = pytesseract.image_to_string(component, config=self.config)
        boxes = self.get_text_boxes(component)
        return extracted_text, boxes


    def get_text_boxes(self, image):
        """Parses Tesseract output to get bounding boxes for detected text."""
        d = pytesseract.image_to_data(image, output_type=Output.DICT)
        boxes = []
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                boxes.append((x, y, x + w, y + h))
        return boxes
