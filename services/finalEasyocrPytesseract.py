import cv2
from matplotlib import pyplot as plt
from services.easyocrExtract import EasyocrExtract
from services.pytesseractExtract import PytesseractExtract

def main():
    img = "../images/text.jpg"
    # Ask user for the extraction method
    extraction_method = input("Choose extraction method (1 for EasyOCR, 2 for Pytesseract): ")

    if extraction_method == "1":
        # Create an instance of EasyOcrExtract class
        extractor = EasyocrExtract()

        show_boxes = input("Show bounding boxes? (y/n): ").lower() == 'y'
        if show_boxes:
            extractor.draw_ocr_boxes(img, "Extracted_Text")
            plt.show()
        else:
            pass
        print("----EXTRECTED TEXT----")
        # Text extraction using EasyOCR
        extractor.extract_text(img)

    elif extraction_method == "2":
        img1= cv2.imread(img)
        # Ask user whether to show bounding boxes
        show_boxes = input("Show bounding boxes? (y/n): ").lower() == 'y'
        text_extractor = PytesseractExtract()
        # Extract text and bounding boxes
        text_regions = text_extractor.extract_text(img1)
        # Access extracted text and bounding boxes
        extracted_text = text_regions[0][0]  # Assuming you want the first region's text
        boxes = text_regions[0][1]  # Assuming you want the first region's bounding boxes
        # Display bounding boxes if chosen
        if show_boxes:
            for box in boxes:
                x, y, x2, y2 = box
                img = cv2.rectangle(img1, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Extracted Text', img)
            cv2.waitKey(0)

        print("Extracted Text:", extracted_text)
    else:
        print("Invalid extraction method selected.")
        return
if __name__ == "__main__":
    main()

