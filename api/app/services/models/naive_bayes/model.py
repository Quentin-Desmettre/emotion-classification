from collections import Counter
import numpy as np
from typing import Callable

"""
Multi-class, Uni-Label N-gram Naive Bayes Classifier
"""
class NBClassifier:
    """
    n: int The n-gram size
    labels: list[str] The list of labels
    pre_processor: Callable[[str], list[str]] The pre-processing function
    laplace_constant: int The Laplace smoothing constant
    alpha: float The alpha parameter for the reweighting of prior probabilities
    """
    def __init__(self, *, n, labels: list[str], pre_processor: Callable[[str], list[str]], laplace_constant=1, alpha=1):
        self.n = n
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.index_to_label = {index: label for index, label in enumerate(labels)}
        self.label_count_for_ngrams = Counter()
        self.label_percentage_for_ngrams = Counter()
        self.pre_processor = pre_processor
        self.prior = None

        # hyper parameters
        self.laplace_constant = laplace_constant
        self.log_laplace_constant = np.log(laplace_constant)
        self.alpha = alpha

    """
    Generate n-grams from a sentence
    """
    def ngrams(self, sentence: list[str]):
        for i in range(len(sentence) - self.n + 1):
            yield tuple(sentence[i:i + self.n])

    """
    Count the number of times each n-gram appears in each label
    """
    def get_label_counts_for_ngrams(self, pre_processed_dataset: list[dict[str: list[str] | int]]) -> Counter:
        label_count_for_ngrams = {}
        for example in pre_processed_dataset:
            for ngram in self.ngrams(example["tokens"]):
                if ngram not in label_count_for_ngrams:
                    label_count_for_ngrams[ngram] = {emotion: 0 for emotion in self.labels}
                emotion_index = example["label"]
                emotion = self.index_to_label[emotion_index]
                label_count_for_ngrams[ngram][emotion] += 1
        return label_count_for_ngrams

    """
    Convert counts to probabilities
    """
    def counts_to_probabilities(self, label_count_for_ngrams: Counter) -> Counter:
        label_prob_for_ngrams = Counter()
        for ngram, label_counts in label_count_for_ngrams.items():
            total = sum(label_counts.values())
            label_prob_for_ngrams[ngram] = {label: count / total for label, count in label_counts.items()}
        return label_prob_for_ngrams

    """
    Calculate the probability of an n-gram appearing in a label, from the counts of n-grams in each label
    """
    def proba_for_ngram_in_label(self, ngram, label: str):
        # Handle missing ngrams with Laplace smoothing
        if ngram not in self.label_count_for_ngrams:
            return self.laplace_constant / len(self.labels)

        # For seen ngrams, apply Laplace smoothing
        total_count = sum(self.label_count_for_ngrams[ngram].values())
        smoothed_count = self.label_count_for_ngrams[ngram][label] + self.laplace_constant
        return smoothed_count / (total_count + len(self.labels))

    """
    Calculate the probability of a sentence appearing in a label, by multiplying the probabilities of each n-gram
    """
    def prob_for_sentence_in_label(self, sentence: list[str], label: str):
        log_prob = 0
        for ngram in self.ngrams(sentence):
            prob = self.proba_for_ngram_in_label(ngram, label)
            if prob > 0:
                log_prob += np.log(prob)
            else:
                log_prob += (self.log_laplace_constant - np.log(len(self.labels)))
        return np.exp(log_prob)

    def get_emotion_counts(self, dataset):
        emotion_counts = {}
        for example in dataset:
            for emotion_index in [example["label"]]:
                emotion = self.index_to_label[emotion_index]
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                emotion_counts[emotion] += 1
        # sort the emotions in the same order as the labels list
        emotion_counts = {emotion: emotion_counts[emotion] for emotion in self.labels}
        return emotion_counts

    """
    Calculate the prior probabilities of each label
    """
    def get_prior(self, pre_processed_dataset: list[dict[str: str | list[int]]]):
        label_counts = self.get_emotion_counts(pre_processed_dataset)
        total_examples = sum(label_counts[example] for example in label_counts)
        emotion_class_counts = {emotion: label_counts[emotion] for emotion in self.labels}
        return {emotion: count / total_examples for emotion, count in emotion_class_counts.items()}

    def proba_for_sentence_for_each_emo_ngram(self, sentence: list[str]) -> dict[str, float]:
        probas = {label: self.prob_for_sentence_in_label(sentence, label) for label in self.labels}
        # Reweight by prior probabilities
        probas = {label: (prob ** (1 - self.alpha)) * (self.prior[label] ** self.alpha) for label, prob in probas.items()}
        # Normalize to sum to 1
        total_proba = sum(probas.values())
        if total_proba > 0:
            probas = {label: prob / total_proba for label, prob in probas.items()}
        return probas

    def predict_label(self, sentence: str) -> dict[str, float]:
        pre_processed_sentence = self.pre_processor(sentence)
        return self.proba_for_sentence_for_each_emo_ngram(pre_processed_sentence)

    def __call__(self, sentence: str):
        return self.predict_label(sentence)

    def train(self, pre_processed_dataset: list[dict[str: list[str] | int]]):
        self.label_count_for_ngrams = self.get_label_counts_for_ngrams(pre_processed_dataset)
        self.label_percentage_for_ngrams = self.counts_to_probabilities(self.label_count_for_ngrams)
        self.prior = self.get_prior(pre_processed_dataset)
