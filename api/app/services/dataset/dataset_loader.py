from datasets import load_dataset
import random

def split_dataset(dataset, train_ratio: float = 0.8,
                  test_ratio: float = 0.1,
                  val_ratio: float = 0.1):
    length = len(dataset["train"])
    # shuffle the dataset
    dataset["train"] = dataset["train"].shuffle()
    train_set = dataset["train"].select(range(int(length * train_ratio)))
    test_set = dataset["train"].select(range(int(length * train_ratio), int(length * (train_ratio + test_ratio))))
    val_set = dataset["train"].select(range(int(length * (train_ratio + test_ratio)), length))

    return train_set, test_set, val_set

def load_twitter_dataset(split = "split"):
    if split not in ["split", "unsplit"]:
        raise ValueError("split must be either 'split' or 'unsplit'")
    ds = load_dataset("dair-ai/emotion", split)

    # Get the train, validation and test sets
    train_set, test_set, val_set = None, None, None

    if split == "split":
        train_set = ds["train"]
        test_set = ds["test"]
        val_set = ds["validation"]
    else:
        # Split the training, validation and test sets, because the "unsplit" dataset only has a training set
        train_set, test_set, val_set = split_dataset(ds)

    # Make sure the dataset is in the right format and in memory
    train_set = train_set.map(lambda x: {"text": x["text"], "label": x["label"]})
    test_set = test_set.map(lambda x: {"text": x["text"], "label": x["label"]})
    val_set = val_set.map(lambda x: {"text": x["text"], "label": x["label"]})

    emotions = [
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    ]


    return train_set, test_set, val_set, emotions

def get_emotions(dataset: str):
    if dataset == "twitter":
        return [
            "sadness",
            "joy",
            "love",
            "anger",
            "fear",
            "surprise",
        ]
    elif dataset == "goemotions":
        return [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
        ]
    else:
        raise ValueError("dataset must be either 'twitter' or 'goemotions'")


def load_goemotions_dataset(split = "simplified"):
    if split not in ["simplified", "raw"]:
        raise ValueError("split must be either 'simplified' or 'raw'")
    emotions = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]
    ds = load_dataset("google-research-datasets/go_emotions", split)

    train_set, test_set, val_set = None, None, None

    if split == "simplified":
        # Get the train and test sets
        train_set = ds["train"]
        test_set = ds["test"]
        val_set = ds["validation"]

        train_set = train_set.map(lambda x: {"text": x["text"], "label": x["labels"][random.randint(0, len(x["labels"])-1)]})
        test_set = test_set.map(lambda x: {"text": x["text"], "label": x["labels"][random.randint(0, len(x["labels"])-1)]})
        val_set = val_set.map(lambda x: {"text": x["text"], "label": x["labels"][random.randint(0, len(x["labels"])-1)]})
    else:
        # Get the train, validation and test sets
        train_set, test_set, val_set = split_dataset(ds)

        # the google dataset raw labels are not stored in a list, but rather there are multiple keys: 'admiration', 'amusement', etc.
        # where if it is present, the value is 1, otherwise it is 0
        # so we need to convert this to a list of labels
        def row_to_label(row):
            for i in range(len(emotions)):
                if row[emotions[i]] == 1:
                    return i


        # Make sure the dataset is in the right format and in memory
        train_set = train_set.map(lambda x: {"text": x["text"], "label": row_to_label(x)})
        test_set = test_set.map(lambda x: {"text": x["text"], "label": row_to_label(x)})
        val_set = val_set.map(lambda x: {"text": x["text"], "label": row_to_label(x)})

    # remove rows that have no label
    # train_set = train_set.filter(lambda x: x["label"] != None)
    # test_set = test_set.filter(lambda x: x["label"] != None)
    # val_set = val_set.filter(lambda x: x["label"] != None)

    return train_set, test_set, val_set, emotions
