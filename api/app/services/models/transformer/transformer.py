from app.types.model import Model
from .model import TransformerModel
from ..lstm.predictor import LstmPredictor


def create() -> Model:
    print("================= Creating Transformer model =================")
    model = TransformerModel.load("cache/transformer_model")
    predictor = LstmPredictor(model, "twitter")
    print("================= Transformer model created =================")
    return predictor
