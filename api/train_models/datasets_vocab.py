import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from app.services.vocab.vocabulary import ModelVocabulary
import os
from app.services.dataset.dataset_loader import load_twitter_dataset, load_goemotions_dataset

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab: ModelVocabulary):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

        self.processed_texts = []
        # with Pool() as p:
        #     self.processed_texts = p.map(lambda x: vocab.tensor_encode(x), texts)
        for text in tqdm(texts):
            self.processed_texts.append(vocab.tensor_encode(text))
        self.processed_labels = torch.tensor(labels, dtype=torch.long, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.processed_texts[idx],
            "labels": self.processed_labels[idx],
        }

def getDatasetsAndVocab(batch_size: int = 32):
    print("Loading dataset...")
    train_set, test_set, val_set, __emotions = load_twitter_dataset("split")
    # google_train_set, google_test_set, google_val_set, google__emotions = load_goemotions_dataset("simplified")
    # train_set, test_set, val_set, __emotions = google_train_set, google_test_set, google_val_set, google__emotions
    print("Dataset loaded")

    EMOTION_TO_INDEX = {
        emotion: index for index, emotion in enumerate(__emotions)
    }

    INDEX_TO_EMOTION = {
        index: emotion for emotion, index in EMOTION_TO_INDEX.items()
    }



    if os.path.exists("cache/twitter_train_vocab.pkl"):
        print("Loading vocabulary from disk...")
        vocab = ModelVocabulary.load("cache/twitter_train_vocab.pkl")
    else:
        print("Building vocabulary...")
        vocab = ModelVocabulary(train_set["text"], 15000, 50)
        vocab.save("cache/twitter_train_vocab.pkl")
    print("Vocabulary built")

    # Prepare datasets
    print("Preparing datasets...")
    if os.path.exists("cache/lstm_train_dataset.pkl"):
        print("Loading datasets from disk...")
        train_dataset = EmotionDataset.load("cache/lstm_train_dataset.pkl")
        val_dataset = EmotionDataset.load("cache/lstm_val_dataset.pkl")
        test_dataset = EmotionDataset.load("cache/lstm_test_dataset.pkl")
    else:
        print("Creating datasets...")
        train_dataset = EmotionDataset(train_set["text"], train_set["label"], vocab)
        val_dataset = EmotionDataset(val_set["text"], val_set["label"], vocab)
        test_dataset = EmotionDataset(test_set["text"], test_set["label"], vocab)

        train_dataset.save("cache/lstm_train_dataset.pkl")
        val_dataset.save("cache/lstm_val_dataset.pkl")
        test_dataset.save("cache/lstm_test_dataset.pkl")
    print("Datasets prepared")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab, EMOTION_TO_INDEX, INDEX_TO_EMOTION
