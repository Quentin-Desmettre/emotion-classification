from app.types.model import Model
from .model import LSTM
from .predictor import LstmPredictor


def create() -> Model:
    print("================= Creating LSTM model =================")
    model = LSTM.load("cache/lstm_model.pth")
    predictor = LstmPredictor(model, "twitter")
    print("================= LSTM model created =================")
    return predictor
