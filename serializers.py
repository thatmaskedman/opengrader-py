import json
from dataclasses import dataclass, asdict, fields
from typing import Dict, Protocol, Any 

class JSONDataclassAdapter:
    def __init__(self, dataclass_instance=None, many: None | list[Any] = None):
        self.dataclass_instance = dataclass_instance
        self.many = many

    def dumps(self) -> str:
        if self.many is not None:
            return json.dumps(
                [asdict(item) for item in self.many]
            )
        return json.dumps(asdict(
            self.dataclass_instance
        ))
        
@dataclass
class Exam:
    exam_group: int
    key_sheet: int
    id: int = None
    name: str = ""
    control_number: str = ""
    correct_answers: int = 0
    wrong_answers: int = 0
    is_graded: bool = False


@dataclass
class Question:
    exam: int
    number: int
    graded_exam: int = 0 
    chosen: bool = False  
    correct: bool = False
    id: int = None 
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
class KeySheet:
    key_class: str 
    exam_group: int
    id: int = None
