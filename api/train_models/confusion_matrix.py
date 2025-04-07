import torch
import torch.nn as nn
import torch.optim as optim
from api.train_models.datasets_vocab import getDatasetsAndVocab
from app.services.models.lstm.model import LSTM

BATCH_SIZE = 32
train_loader, val_loader, test_loader, vocab, EMOTION_TO_INDEX, INDEX_TO_EMOTION = getDatasetsAndVocab(BATCH_SIZE)
__emotions = list(EMOTION_TO_INDEX.keys())

NUM_CLASSES = len(EMOTION_TO_INDEX.keys())

print("Loading model...")
model = LSTM.load("cache/lstm_model.pth")
print("Model built")

from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

def get_model_performance(model, data_loader, index_to_emotion, device):
    """
    Evaluate the model and compute accuracy per emotion.

    Args:
        model: The trained model.
        data_loader: DataLoader for the evaluation set.
        index_to_emotion: A dictionary mapping label indices to emotion names.
        device: The device to run the evaluation on.

    Returns:
        A dictionary with emotion-wise accuracy and a plot.
    """
    model.eval()
    emotion_correct = defaultdict(int)
    emotion_total = defaultdict(int)
    total_predictions = defaultdict(int)

    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            # Update emotion-specific counts
            for true_label, pred_label in zip(labels, preds):
                true_emotion = index_to_emotion[true_label]
                predicted_emotion = index_to_emotion[pred_label]
                emotion_total[true_emotion] += 1
                true_labels.append(true_emotion)
                predicted_labels.append(predicted_emotion)
                total_predictions[true_emotion] += 1
                if true_emotion == predicted_emotion:
                    emotion_correct[true_emotion] += 1
                else:
                    false_positives[predicted_emotion] += 1
                    false_negatives[true_emotion] += 1

     # Calculate accuracy, recall, precision and F1 score for each emotion
    metrics = {}
    for label in __emotions:
        tp = emotion_correct[label]
        fp = false_positives[label]
        fn = false_negatives[label]

        # Calculate precision
        precision = tp / (tp + fp) if tp + fp > 0 else 0

        # Calculate recall
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        metrics[label] = {
            "accuracy": emotion_correct[label] / total_predictions[label],
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    # Calculate total accuracy
    total_accuracy = sum(emotion_correct.values()) / sum(total_predictions.values())

    return metrics, total_accuracy, true_labels, predicted_labels

def evaluate_model(model, data_loader, index_to_emotion, device, title="Model Performance"):
    metrics, total_accuracy, true_labels, predicted_labels = get_model_performance(
        model, data_loader, index_to_emotion, device
    )

    # Plot accuracy per emotion
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), [metric["accuracy"] * 100 for metric in metrics.values()])
    plt.title("Accuracy per Emotion", fontsize=16)
    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("accuracy_per_emotion.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

    disp = ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels, labels=__emotions, normalize="true", cmap="Blues")
    fig = disp.figure_
    fig.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight", transparent=True)
    # uncomment for a larger plot
    # fig = disp.figure_
    # fig.set_figwidth(40)
    # fig.set_figheight(40)
    # # disp.plot(cmap="Blues")
    # disp.im_.set_clim(0, 1)
    # plt.title(title)
    # plt.xticks(rotation=60)
    # plt.show()

    # Create a table of metrics
    metrics_table = []
    for emotion, metric in metrics.items():
        metrics_table.append([emotion, f"{metric['accuracy']:.2f}", f"{metric['precision']:.2f}", f"{metric['recall']:.2f}", f"{metric['f1_score']:.2f}"])
    metrics_table.append(["Total", f"{total_accuracy:.2f}", "", "", ""])
    plt.figure(figsize=(15, 10))
    plt.axis('off')
    # plot table, adding a little height padding
    plt.table(cellText=metrics_table, colLabels=["Emotion", "Accuracy", "Precision", "Recall", "F1 Score"], cellLoc="center", loc="center")
    plt.title(title, fontsize=15)
    plt.show()

    # Print the results
    print(f"Overall Accuracy: {total_accuracy:.2f}%")
    for emotion, metric in metrics.items():
        print(f"{emotion.capitalize()}:")
        print(f"  - Accuracy: {metric['accuracy']:.2f}")
        print(f"  - Precision: {metric['precision']:.2f}")
        print(f"  - Recall: {metric['recall']:.2f}")
        print(f"  - F1 Score: {metric['f1_score']:.2f}")
        print()

    return total_accuracy

# Evaluate the trained model
evaluate_model(
    model=model,
    data_loader=test_loader,
    index_to_emotion=INDEX_TO_EMOTION,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
