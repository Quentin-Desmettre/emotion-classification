from collections import Counter
from multiprocessing import Pool
from spacy.tokens import Doc
import spacy
import pickle
import torch

class ModelVocabulary:
    def __init__(self, texts: list[str], max_vocab_size: int, max_len: int):
        self.nlp = spacy.load("en_core_web_md")
        self.vocab, self.preprocessed_texts = self._build_vocab(texts, max_vocab_size)
        self.max_len = max_len
        self.vocab_size = len(self.vocab)

    def preprocess(self, text: str) -> list[str]:
        return self.preprocess_doc(self.nlp(text))

    def preprocess_doc(self, doc: Doc) -> list[str]:
        return [token.lemma_.lower().strip() for token in doc if not token.like_url and not token.is_stop]

    def preprocess_batch(self, texts: list[str]) -> list[list[str]]:
        processed_texts = []
        for doc in self.nlp.pipe(texts, batch_size=1024):  # Adjust batch_size based on memory
            processed_texts.append(self.preprocess_doc(doc))
            if len(processed_texts) % 1000 == 0:
                print("Processed", len(processed_texts), "/", len(texts))
        print("Processed", len(processed_texts), "/", len(texts))
        return processed_texts

    def _build_vocab(self, texts: list[str], max_vocab_size: int) -> tuple[dict[str, int], list[list[str]]]:
        """
        Builds a vocabulary mapping tokens to unique indices from a list of texts.
        - The vocabulary is constructed based on token frequency, keeping the most common tokens
        up to the specified `max_vocab_size`.
        - Tokens beyond `max_vocab_size` are excluded from the vocabulary.

        Args:
            texts (list[str]): A list of input texts to be tokenized and processed.
            max_vocab_size (int): The maximum size of the vocabulary, including special tokens.

        Returns:
            dict[str, int]: A dictionary where:
                - Keys are tokens (words).
                - Values are unique integer indices.
                Special tokens included:
                - "<PAD>": 0 (used for padding).
                - "<UNK>": 1 (used for unknown tokens).
                Other tokens are assigned indices starting from 2.


        """
        # with Pool() as p:
        #     preprocessed_texts = p.map(self.preprocess_doc, self.nlp.pipe(texts, batch_size=1024))
        #     counter = Counter(token for tokens in preprocessed_texts for token in tokens)
        preprocessed_texts = []
        for text in self.nlp.pipe(texts, batch_size=1024):
            preprocessed_texts.append(self.preprocess_doc(text))
        counter = Counter(token for tokens in preprocessed_texts for token in tokens)

        most_common: list[tuple[str, int]] = counter.most_common(max_vocab_size - 2)
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1
        return vocab, preprocessed_texts

    def _encode(self, text: str) -> list[int]:
        """
        Encodes a text into a fixed-length sequence of token IDs using a given vocabulary.

        Args:
            text: The input text to be tokenized and encoded.
            vocab (dict[str, int: A dictionary mapping tokens to their corresponding IDs.
                Special tokens:
                - "<PAD>": ID used for padding.
                - "<UNK>": ID used for unknown tokens.
            max_len (int): The maximum length of the encoded sequence. Longer sequences are truncated,
                and shorter sequences are padded.

        Returns:
            list[int]: A list of token IDs representing the encoded text, with a fixed length of `max_len`.
        """

        tokens = self.preprocess(text)
        ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        if len(ids) > self.max_len:
            return ids[:self.max_len]
        return ids + [self.vocab["<PAD>"]] * (self.max_len - len(ids))

    def tensor_encode(self, text: str, *, gpu: bool = True) -> torch.Tensor:
        base = torch.tensor(self._encode(text), dtype=torch.long)
        # put in GPU
        if gpu and torch.cuda.is_available():
            base = base.to(torch.device("cuda"))
        return base

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ModelVocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)
