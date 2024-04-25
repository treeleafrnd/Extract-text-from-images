import cv2
from services.pytesseractExtract import PytesseractExtract
image_path = '../images/text.jpg'
image = cv2.imread(image_path)
def main():
    # Create TextExtractor instance
    text_extractor = PytesseractExtract()

    extracted_text, boxes = text_extractor.extract_text(image)

    # Print extracted text
    print("Extracted Text:")
    print(extracted_text)

    # Draw bounding boxes
    for box in boxes:
      x, y, x2, y2 = box
      img = cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Extracted Text', image)
    cv2.waitKey(0)

if __name__=="__main__":
  main()
