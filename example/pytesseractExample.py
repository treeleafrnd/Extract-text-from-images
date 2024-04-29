import cv2
from services.pytesseractExtract import PytesseractExtract
image_path = '../images/sample9.png'
image = cv2.imread(image_path)
def main():
    # Create TextExtractor instance
    text_extractor = PytesseractExtract()
    # Extract text and bounding boxes
    text_regions = text_extractor.extract_text(image)

    # Access extracted text and bounding boxes
    extracted_text = text_regions[0][0]  # Assuming you want the first region's text
    boxes = text_regions[0][1]  # Assuming you want the first region's bounding boxes

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
