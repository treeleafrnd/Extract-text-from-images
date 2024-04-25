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
        extractor = PytesseractExtract()
        # Extract text
        extracted_text, boxes = extractor.extract_text(img1)

        # Display bounding boxes if chosen
        if show_boxes:
            for box in boxes:
                cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title("Text with bounding boxes")
            plt.axis('off')
            plt.show()

        print("Extracted Text:", extracted_text)
    else:
        print("Invalid extraction method selected.")
        return
if __name__ == "__main__":
    main()

