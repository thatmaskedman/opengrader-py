from array import ArrayType
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from cv2 import cv

@dataclass
class Answer:
    """
    """
    letter: str 
    contour: ArrayType
    filled: bool = False 
    threshold: float = 0.0
    
@dataclass
class Question:
    number: int
    answers: dict[str, Answer] = {
        'a': Answer('a'),
        'b': Answer('b'),
        'c': Answer('c'),
        'd': Answer('d'),
        'e': Answer('e'),    
    }

@dataclass
class KeySheet:
    keytype: str
    keys: dict[int, str]

    def serialize() -> str:
        pass

class AnswersSheet:
    def __init__(self, contours: np.ndarray) -> None:
        self.questions: list[Question] = [
            Question(n) 
            for n in range(1,51)
        ]
        self.contours: np.ndarray = contours
        self.key_sheets: dict[str, KeySheet] = {
            'a': KeySheet('a'),
            'b': KeySheet('b'),
            'c': KeySheet('c'),
            'd': KeySheet('d'),
    }

    def set_contours(self, conts: np.ndarray) -> None:
        conts = conts.reshape((-1, 5))
        for question in self.questions:
            for k, v in zip(question.answers.keys(), conts):
                question.answers[k].contour = v

    