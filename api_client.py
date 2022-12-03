import requests
import serializers
from dataclasses import asdict
import json

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url 

    def post_exam(self, exam: serializers.Exam, questions: list[serializers.Question]):
        res: requests.Response
        exam_fields = exam.as_dict()
        print(exam_fields)
    
        res = requests.post(
            f'{self.base_url}/exams/', 
            data=exam_fields)

        exam_id: int = res.json().get('id')
        for q in questions:
            q.graded_exam = exam_id
            print(q.as_dict())

        
        questions_field_list = [q.as_dict() for q in questions]
        # print(questions_field_list)
        print(json.dumps(questions_field_list))
        res = requests.post(
            f'{self.base_url}/questions/', 
            json=questions_field_list)

        # print(res.json())