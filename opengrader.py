import cv2 as cv
import numpy as np
from document import DocumentProcessor
def main():
    img = cv.imread('sheet/answered.jpg')
    processor = DocumentProcessor(img)
    processor.process()
    processor._write_steps()


if __name__ == '__main__':
    main()