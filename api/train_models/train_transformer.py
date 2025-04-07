import torch
import torch.nn as nn
import torch.optim as optim
from .datasets_vocab import getDatasetsAndVocab
from app.services.models.transformer.model import TransformerModel
from app.services.models.lstm.trainer import LstmTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
train_loader, val_loader, test_loader, vocab, EMOTION_TO_INDEX, _ = getDatasetsAndVocab(BATCH_SIZE)

NUM_CLASSES = len(EMOTION_TO_INDEX.keys())


MAX_LEN = 128
EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
model = TransformerModel(
    vocab_size=vocab.vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    max_len=MAX_LEN,
    dropout=DROPOUT,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.01)

print("Model built")
print("Training model...")
trainer = LstmTrainer(model, train_loader, val_loader, optimizer, True, criterion)
best_model = trainer.train(epochs=30, verbose=True, doEvaluate=True)
print("Model trained")
# Save the model
best_model.save("cache/transformer_model")
print("Model saved")
