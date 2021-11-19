import json


class Explanation:
    def __init__(self, tokens: list, attributions: list, classification: float):
        self.tokens = tokens
        self.attributions = attributions
        self.classification = classification

    def to_json(self):
        return json.dumps({"tokens": self.tokens, "attributions": list(self.attributions), "classification": self.classification})

    @staticmethod
    def from_json(string: str):
        d = json.loads(string)
        return Explanation(tokens=d['tokens'], attributions=d['attributions'], classification=d['classification'])

