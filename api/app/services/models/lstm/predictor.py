from app.services.vocab.vocabulary import ModelVocabulary
from .model import LSTM
from app.services.dataset.dataset_loader import get_emotions, load_twitter_dataset, load_goemotions_dataset
from app.services.vocab.vocabulary import ModelVocabulary
import os, torch

class LstmPredictor:
    def __init__(self, model: LSTM, type: str = "twitter"):
        self.emotions = get_emotions(type)
        vocab_path = "cache/" + ("twitter_train_vocab.pkl" if type == "twitter" else "goemotions_train_vocab.pkl")
        vocab = None
        if os.path.exists(vocab_path):
            print("Loading vocabulary from disk...")
            vocab = ModelVocabulary.load(vocab_path)
        else:
            train_set, _, _, _ = load_twitter_dataset("split") if type == "twitter" else load_goemotions_dataset("simplified")
            print("Creating vocabulary...")
            vocab = ModelVocabulary(train_set["text"], 15000, 50)
            vocab.save(vocab_path)
        self.vocab = vocab
        self.model = model

    def __call__(self, text: str):
        encoded = self.vocab.tensor_encode(text)
        encoded = encoded.unsqueeze(0)
        if torch.cuda.is_available():
            encoded = encoded.to(torch.device("cuda"))
        output = self.model(encoded)

        # apply softmax to output
        output = torch.nn.functional.softmax(output, dim=1)
        # return dictionary of emotions and their probabilities
        return {emotion: output[0][i].item() for i, emotion in enumerate(self.emotions)}
