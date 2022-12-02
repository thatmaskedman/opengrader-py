from typing import Any, TypedDict
import json
import cv2 as cv
import numpy as np
import numpy.typing as npt
import itertools as it
import requests
import serializers
from dataclasses import dataclass, field
# from collections import namedtuple
# from cv2 import cv

class QuestionData(TypedDict):
    number: int
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

class QuestionDatabaseFields(TypedDict):
    graded_exam: int

class AnswerSheet:
    _BGR_RED = (0,0,255)
    _BGR_YELLOW = (0,255,255)
    _BGR_GREEN = (0,255,0)

    def __init__(
            self, 
            img, 
            points: npt.NDArray[np.int32],  
            thresholds: npt.NDArray[np.float64], 
            question_count: int, 
            name: str = '',
            control_num: str = '',
            exam_group: int = None) -> None:

        self.img = img
        self_question_data: QuestionData = {}
        self.name = name
        self.control_num = control_num
        self_keysheet_data: dict[Any, Any] = {}
        self.name_img = np.array([])
        self.control_num_img = np.array([])
        self.question_count = 50
        self.correct_count = 0
        self.ratio = 0.0
        self.thresholds = thresholds
        self.points: npt.NDArray[np.int32] = points
        

    def set_data(self) -> None:
        points = self.points.reshape((-1,5,2))
        intensities = self.thresholds.reshape((-1,5))
        
        fields: list[QuestionData] = [] 
        for index, item in enumerate(zip(points, intensities), 1):
            point, intensity = item
            question: QuestionData = {}
            for p, i, letter in zip(point, intensity, it.cycle('abcde')):
                i = float(i)
                question.update({
                    'number': index, 
                    f'{letter}_filled': i != 0.0,
                    f'{letter}_thresh': i
                }) 
            fields.append(question)

        self.question_data = fields

    #FIXME
    def grade(self):
        with open('keys/dummykey.json', 'r') as f:
            keys = json.load(f)
        dummy_key = KeySheet('a', keys)
        print(len(keys))

        mykey = {q.number: q.chosen for q in self.questions}

        for q in self.questions:
            q.correct = dummy_key.keys[q.number] == mykey[q.number]
        
        self.correct_count = len(list(filter(lambda q: q.correct, self.questions)))
        self.ratio = self.correct_count / self.question_count

        print(self.correct_count)
        print(self.ratio)

    def mark_grade(self):
        for q in filter(lambda q: q.correct, self.questions):
            center = q.answers[q.chosen].point
            cv.circle(self.img, center, 12, self._BGR_GREEN, 2)
        
        for q in filter(lambda q: not q.correct, self.questions):
            if q.chosen != '':
                center = q.answers[q.chosen].point
                cv.circle(self.img, center, 12, self._BGR_RED, 2)

        # for zip(points, intensities)

        # print(points)
        # print(intensities)
        # for q, p in zip(self.questions, points):
            # pass 
        # for question in self.questions:
        #     for k, v in zip(question.answers.keys(), points):
        #         question.answers[k].point = v