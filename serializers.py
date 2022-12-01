import json
from dataclasses import dataclass, asdict

class Serializable:
    def as_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class Exam(Serializable):
    exam_group: int
    key_sheet: int
    id: int = None
    name: str = ""
    control_number: str = ""
    correct_answers: int = 0
    wrong_answers: int = 0
    is_graded: bool = False


@dataclass
class Question(Serializable):
    exam: int
    number: int
    graded_exam: int = 0 
    chosen: bool = False  
    correct: bool = False
    id: int = None 
    threshold: float = 0.0
    a_filled: bool = False 
    b_filled: bool = False 
    c_filled: bool = False 
    d_filled: bool = False 
    e_filled: bool = False 


@dataclass
class KeySheet(Serializable):
    key_class: str 
    exam_group: int
    id: int = None
