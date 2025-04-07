from .models.lstm import lstm
from .models.naive_bayes import naive_bayes
from .models.transformer import transformer
from app.types.model import Model
from typing import Callable

def load_models() -> dict[str, Model]:
    return {
        "naive_bayes": naive_bayes.create(),
        "lstm": lstm.create(),
        "transformer": transformer.create(),
    }
