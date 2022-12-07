#!/bin/python3

import argparse
import cv2 as cv
# import numpy as np
from cvprocessor.answersheet import AnswerSheet
from cvprocessor.document import DocumentProcessor
from restful.api_client import APIClient
import common.serializers as serializers
import json
import io


def main():
    parser = argparse.ArgumentParser(
        description='Opengrader CLI Tool'
    )

    parser.add_argument(
        '-e', '--exam', help='Path to the exam answer sheet.', required=True)
    parser.add_argument('-k', '--keysheet',
                        help='Path to the Key Sheet JSON answer file.')

    parser.add_argument('--grade', help='Attempt to grade.')
    parser.add_argument('--out', help='Specify output directory.')
    parser.add_argument('--url', help='Base URL')
    args = parser.parse_args()

    if args.exam:
        question_count = 50
        img_bytes: io.BytesIO = None
        with open(args.exam, 'rb') as f:
            img_bytes = io.BytesIO(f.read())

        processor = DocumentProcessor(img_bytes)
        processor.process()
        processor._write_steps()

        points = processor.get_choice_points()
        intensities = processor.get_intensities()

        answer_sheet = AnswerSheet(
            processor.img_scaled, points, intensities, question_count)
        answer_sheet.set_data()
        # print(answer_sheet.question_data)
        answer_sheet.choose_answers()
        # print(json.dumps(answer_sheet.question_data))
        # print(answer_sheet.)

        if args.keysheet:
            with open(args.keysheet, 'r') as f:
                key_data: dict[str, str] = json.load(f)

            answer_sheet.set_keydata(key_data)
            answer_sheet.grade_data()
            print(json.dumps(answer_sheet.question_data))
            answer_sheet.mark_choices()
            cv.imwrite('graded.jpg', answer_sheet.img)

        if args.url:
            exam = serializers.Exam(
                1, 1, name="John Doe", control_number="17330462")

            questions = [serializers.Question(
                **fields) for fields in answer_sheet.question_data]

            client = APIClient(args.url)
            client.post_exam(exam=exam, questions=questions)

        # answer_sheet.grade()
        # answer_sheet.mark_grade()
        # AnswerSheet()


if __name__ == '__main__':
    main()
