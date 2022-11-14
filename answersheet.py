from typing import Any
import json
import cv2 as cv
import numpy as np
import numpy.typing as npt
import itertools as it
import urllib.request
from dataclasses import dataclass, field
# from collections import namedtuple
# from cv2 import cv

@dataclass
class Choice:
    """
    """
    letter: str
    number: int
    intensity: float = 0.0
    coord: str = field(init=False)
    point: npt.NDArray[np.int32] = np.array([])
    filled: bool = False 

    def __post_init__(self):
        self.coord = f'{self.number}{self.letter}'
    
    
@dataclass
class Question:
    number: int
    answers: dict[str, Choice] = field(default_factory=dict[str, Choice])
    chosen: str = ''
    correct: bool = False

@dataclass
class KeySheet:
    keytype: str
    keys: dict[int, str] = field(default_factory=dict[int, str])

    # def serialize() -> str:
    #     return ""

class AnswerSheet:
    _BGR_RED = (0,0,255)
    _BGR_YELLOW = (0,255,255)
    _BGR_GREEN = (0,255,0)

    def __init__(self, img, points: npt.NDArray[np.int32], intensities: npt.NDArray[np.float64]) -> None:
        self.img = img
        self.name = ''
        self.control_num = ''
        self.img = img
        self.name_img = np.array([])
        self.control_num_img = np.array([])
        self.questions: list[Question] = [
            Question(n) 
            for n in range(1,51)
        ]
        self.question_count = 50
        self.correct_count = 0
        self.ratio = 0.0
        self.intensities = intensities
        self.points: npt.NDArray[np.int32] = points
        self.key_sheets: dict[str, KeySheet] = {
            'a': KeySheet('a'),
            'b': KeySheet('b'),
            'c': KeySheet('c'),
            'd': KeySheet('d'),
            'e': KeySheet('e'),
        }

        self.questions_json = ''
        self.graded_json = ''
        
    def grade(self):
        dummy_key = self.key_sheets.get('a')
        dummy_key.keys = {
            1: 'b',
            2: 'e',
            3: 'c',
            4: 'c',
            5: 'b',
            6: 'c',
            7: 'a',
            8: 'c',
            9: 'b',
            10: 'b',
            11: 'b',
            12: 'a',
            13: 'b',
            14: 'e',
            15: 'a',
            16: 'e',
            17: 'b',
            18: 'a',
            19: 'a',
            20: 'e',
            21: 'a',
            22: 'b',
            23: 'd',
            24: 'e',
            25: 'a',
            26: 'a',
            27: 'e',
            28: 'c',
            29: 'c',
            30: 'a',
            31: 'b',
            32: 'a',
            33: 'b',
            34: 'a',
            35: 'c',
            36: 'a',
            37: 'b',
            38: 'a',
            39: 'a',
            40: 'd',
            41: 'a',
            42: 'a',
            43: 'c',
            44: 'e',
            45: 'd',
            46: 'd',
            47: 'd',
            48: 'd',
            49: 'a',
            50: 'e'
        }

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

    def set_data(self) -> None:
        points = self.points.reshape((-1,5,2))
        intensities = self.intensities.reshape((-1,5))
        
        choices = [] 
        for index, item in enumerate(zip(points, intensities), 1):
            point, intensity = item
            for p, i, letter in zip(point, intensity, it.cycle('abcde')):
                choices.append(Choice(letter, index, i, p, filled=(i != 0.0)))

        questions: list[Question] = []
        for n in range(1,51):
            fields = {
                choice.letter: choice 
                for choice in filter(lambda c: c.number == n, choices)
            }
            questions.append(Question(n, fields))

        for q in questions:
            chosen = list(filter(lambda c: c.filled,  q.answers.values()))
    
            if len(chosen) == 1:
                q.chosen = chosen[0].letter
            
            #FIXME
            elif len(chosen) > 1:
                pass

            print(q, '\n')
        self.questions = questions
        # for zip(points, intensities)

        # print(points)
        # print(intensities)
        # for q, p in zip(self.questions, points):
            # pass 
        # for question in self.questions:
        #     for k, v in zip(question.answers.keys(), points):
        #         question.answers[k].point = v