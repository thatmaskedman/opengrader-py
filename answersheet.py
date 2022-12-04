from typing import Any, TypedDict
# import json
import cv2 as cv
import numpy as np
import numpy.typing as npt
import itertools as it
# import requests
# from dataclasses import dataclass, field


class Examdata(TypedDict):
    name: str
    control_num: str 


class KeySheetData(TypedDict):
    kind: str


class QuestionData(TypedDict):
    number: int
    chosen: str
    a_thresh: float
    b_thresh: float
    c_thresh: float
    d_thresh: float
    e_thresh: float
    a_filled: bool
    b_filled: bool
    c_filled: bool
    d_filled: bool
    e_filled: bool
    correct: bool


class KeyData(TypedDict):
    number: int
    chosen: str


class AnswerSheet:
    _BGR_RED = (0, 0, 255)
    _BGR_YELLOW = (0, 255, 255)
    _BGR_GREEN = (0, 255, 0)

    def __init__(
            self,
            img,
            points: npt.NDArray[np.int32],
            thresholds: npt.NDArray[np.float64], 
            question_count: int,
            name: str = '',
            control_num: str = '', ) -> None:

        self.img = img
        self.img_graded = None
        self.question_data: QuestionData = {}
        self.graded_questions: QuestionData = {}
        self.key_data: KeyData = {}
        self.name = name
        self.control_num = control_num
        self.name_img = np.array([])
        self.control_num_img = np.array([])
        self.question_count = 50
        self.correct_count = 0
        self.ratio = 0.0
        self.thresholds = thresholds
        self.points: npt.NDArray[np.int32] = points

    def set_data(self) -> None:
        points = self.points.reshape((-1, 5, 2))
        intensities = self.thresholds.reshape((-1, 5))

        fields: list[QuestionData] = []
        for index, item in enumerate(zip(points, intensities), 1):
            point, intensity = item
            question: QuestionData = {
                'number': index, 'chosen': '', 'correct': False}
            for p, i, letter in zip(point, intensity, it.cycle('abcde')):
                i = float(i)
                question.update({
                    f'{letter}_filled': i != 0.0,
                    f'{letter}_thresh': i
                })
            fields.append(question)

        self.question_data = fields

    def set_keydata(self, key_data: KeyData):
        self.key_data = key_data

    def choose_answers(self) -> None:
        question: QuestionData
        chosen_questions: list[QuestionData] = []
        choices: dict[str, tuple[float, bool]]
        chosen_candidates: dict[str, tuple[float, bool]]
        
        for question in self.question_data:
            choices = {
                'a': (question['a_thresh'], question['a_filled']),
                'b': (question['b_thresh'], question['b_filled']),
                'c': (question['c_thresh'], question['c_filled']),
                'd': (question['d_thresh'], question['d_filled']),
                'e': (question['e_thresh'], question['e_filled']),
            }

            chosen_candidates = {
                choice: fields for choice, fields in choices.items() if fields[1]
            }
            if len(chosen_candidates) == 1:
                chosen_letter, _ = chosen_candidates.popitem()
                chosen_fields: QuestionData = question.copy()
                chosen_fields['chosen'] = chosen_letter
                chosen_questions.append(chosen_fields)

            elif len(chosen_candidates) > 1:
                chosen_letter, _ = sorted(
                    chosen_candidates.items(),
                    key=lambda c: c[1][0],
                    reverse=True)[0]
                chosen_fields: QuestionData = question.copy()
                chosen_fields['chosen'] = chosen_letter
                chosen_questions.append(chosen_fields)

            else:
                chosen_questions.append(question)

        self.question_data = chosen_questions

    def grade_data(self):
        graded_questions: list[QuestionData] = []
        graded: dict[str, tuple[float, bool]]

        question: QuestionData
        for question, key in zip(self.question_data, self.key_data):
            graded = question.copy()
            graded['correct'] = question['chosen'] == key['chosen']
            graded_questions.append(graded)

        self.question_data = graded_questions

    def mark_choices(self):
        question: QuestionData
        point: npt.ArrayLike
        for question, point in zip(self.question_data, self.points):
            a_choice, b_choice, c_choice, d_choice, e_choice = tuple(point)
            dict[str, Any]
            match question['chosen']:
                case 'a':
                    chosen = a_choice
                case 'b':
                    chosen = b_choice
                case 'c':
                    chosen = c_choice
                case 'd':
                    chosen = d_choice
                case 'e':
                    chosen = e_choice
                case _:
                    chosen = []
            if not any(chosen):
                continue

            if question['correct']:
                cv.circle(self.img, chosen, 12, self._BGR_GREEN, 2)
            else: 
                cv.circle(self.img, chosen, 12, self._BGR_RED, 2)
