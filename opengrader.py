from concurrent.futures import process
import cv2 as cv
import numpy as np
from answersheet import AnswerSheet
# from answersheet import AnswerSheet
from document import DocumentProcessor

def main():
    img = cv.imread('sheet/answered.jpg')
    processor = DocumentProcessor(img)
    processor.process()
    # processor._write_steps()
    
    points = processor.get_choice_points()
    intensities = processor.get_intensities()
    answer_sheet = AnswerSheet(points, intensities)
    answer_sheet.set_data()
    answer_sheet.grade()
    # AnswerSheet()

if __name__ == '__main__':
    main()