import json
from dataclasses import dataclass, asdict, fields
from typing import Dict, Protocol, Any 

class Serializable:
    def as_dict(self) -> dict:
        return {k:v for k, v in asdict(self).items() if v is not None}

    def json_dumps(self) -> str:
        if self.many is not None:
            return json.dumps(
                [asdict(item) for item in self.many]
            )
        return json.dumps(asdict(
            self.dataclass_instance
        ))
        

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
    number: int 
    id: int = None 
    graded_exam: int = None 
    chosen: str = ''  
    correct: bool = False
    threshold: float = 0.0
    a_thresh: float = 0.0 
    b_thresh: float = 0.0 
    c_thresh: float = 0.0 
    d_thresh: float = 0.0 
    e_thresh: float = 0.0 
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
