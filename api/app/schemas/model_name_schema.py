from enum import Enum

class ModelName(str, Enum):
    naive_bayes = "naive_bayes"
    lstm = "lstm"
    transformer = "transformer"
