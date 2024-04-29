import matplotlib.pyplot as plt

from services.easyocrExtract import EasyocrExtract


def main():

  # Replace with your image path
  img = ("../images/text.jpg")
  # Create an instance of EasyOcrExtract class
  extractor = EasyocrExtract()
  # Text extraction using EasyOCR
  extractor.extract_text(img)
  # Optional: Overlay bounding boxes on the image
  extractor.draw_ocr_boxes(img, "Extracted_Text")
  plt.show()

if __name__ == "__main__":
  main()