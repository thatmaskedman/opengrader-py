from concurrent.futures import process
import cv2 as cv
import numpy as np
from answersheet import AnswerSheet
# from answersheet import AnswerSheet
from document import DocumentProcessor

def main():
    question_count = 50

    img = cv.imread('sheet/foobaz.png')
    processor = DocumentProcessor(img)
    processor.process()
    processor._write_steps()
    
    points = processor.get_choice_points()
    intensities = processor.get_intensities()
    
    answer_sheet = AnswerSheet(processor.img_scaled, points, intensities, question_count)
    answer_sheet.set_data()
    # answer_sheet.grade()
    # answer_sheet.mark_grade()
    cv.imwrite('graded.jpg', answer_sheet.img)
    # AnswerSheet()

if __name__ == '__main__':
    main()