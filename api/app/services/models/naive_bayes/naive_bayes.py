from app.types.model import Model
from .model import NBClassifier
from app.services.dataset.dataset_loader import load_twitter_dataset
from app.services.vocab.vocabulary import ModelVocabulary
import os
import pickle

def create() -> Model:
    train_set, _, _, __emotions = load_twitter_dataset("split")
    if os.path.exists("cache/twitter_train_vocab.pkl"):
        print("Loading vocabulary from disk...")
        vocab = ModelVocabulary.load("cache/twitter_train_vocab.pkl")
    else:
        print("Creating vocabulary...")
        vocab = ModelVocabulary(train_set["text"], 15000, 50)
        vocab.save("cache/twitter_train_vocab.pkl")

    model = NBClassifier(
        n=2,
        labels=__emotions,
        pre_processor=lambda x: vocab.preprocess(x),
        laplace_constant=0.8,
        alpha=0.01875,
    )

    if os.path.exists("cache/naive_bayes_preprocessed_train_set.pkl"):
        print("Loading preprocessed train set from disk...")
        pre_processed_train_set = pickle.load(
            open("cache/naive_bayes_preprocessed_train_set.pkl", "rb")
        )
    else:
        print("Preprocessing train set...")
        pre_processed_train_set = [
            {"tokens": tokens, "label": train_set[i]["label"]}
            for i, tokens in enumerate(
                vocab.preprocess_batch([example["text"] for example in train_set])
            )
        ]
        with open("cache/naive_bayes_preprocessed_train_set.pkl", "wb") as f:
            pickle.dump(pre_processed_train_set, f)
    print("Training model...")
    model.train(pre_processed_train_set)
    print("Model trained")
    return model
