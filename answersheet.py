import numpy as np
import numpy.typing as npt
import itertools as it 
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
    answers: dict[str, Choice] = field(init=False) 
    correct: bool = False

    def __post_init__(self) -> None:
        self.answers = {
            'a': Choice('a', self.number),
            'b': Choice('b', self.number),
            'c': Choice('c', self.number),
            'd': Choice('d', self.number),
            'e': Choice('e', self.number),    
        }

@dataclass
class KeySheet:
    keytype: str
    keys: dict[int, str] = field(default_factory=dict[int, str])

    # def serialize() -> str:
    #     return ""

class AnswerSheet:
    _BGR_RED = (0,0,255)
    _BGR_YELLOW = (0,255,255)
    _BGR_GREEN = (0,255,255)

    def __init__(self, points: npt.NDArray[np.int32], intensities: npt.NDArray[np.float64]) -> None:
        self.name = ''
        self.control_num = ''
        self.questions: list[Question] = [
            Question(n) 
            for n in range(1,51)
        ]
        self.intensities = intensities
        self.points: npt.NDArray[np.int32] = points
        self.key_sheets: dict[str, KeySheet] = {
            'a': KeySheet('a'),
            'b': KeySheet('b'),
            'c': KeySheet('c'),
            'd': KeySheet('d'),
            'e': KeySheet('e'),
        }

    def get_key(self):
        for q in self.questions:
            pass
            
        pass

    def grade(self):
        dummy_key = KeySheet('a')
        dummy_key.keys = {
            1: 'a',
            2: 'b',
            3: 'c',
            4: 'a',
            5: 'b',
            6: 'c',
            7: 'a',
            8: 'b',
            9: 'c',
            10: 'c'
        }

    def set_data(self) -> None:
        points = self.points.reshape((-1,5,2))
        intensities = self.intensities.reshape((-1,5))
        
        choices = [] 

        for index, item in enumerate(zip(points, intensities), 1):
            point, intensity = item
            for p, i, letter in zip(point, intensity, it.cycle('abcde')):
                choices.append(Choice(letter, index, i, p, filled= (i != 0.0)))

        for c in choices:
            print(c)
        # for zip(points, intensities)

        # print(points)
        # print(intensities)
        # for q, p in zip(self.questions, points):
            # pass 
        # for question in self.questions:
        #     for k, v in zip(question.answers.keys(), points):
        #         question.answers[k].point = v