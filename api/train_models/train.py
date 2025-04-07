import torch
import torch.nn as nn
import torch.optim as optim
from .datasets_vocab import getDatasetsAndVocab
from app.services.models.lstm.model import LSTM

BATCH_SIZE = 32
train_loader, val_loader, test_loader, vocab, EMOTION_TO_INDEX, _ = getDatasetsAndVocab(BATCH_SIZE)

NUM_CLASSES = len(EMOTION_TO_INDEX.keys())


print("Building model...")
# if os.path.exists("cache/lstm_model.pth"):
#     print("Loading model from disk...")
#     model = LSTM.load("cache/lstm_model.pth")
# else:
#     print("Creating model...")
model = LSTM(embed_dim=32, hidden_dim=64, dropout_p=0.1, VOCAB_SIZE=vocab.vocab_size, NUM_CLASSES=NUM_CLASSES).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.01)
print("Model built")

from app.services.models.lstm.trainer import LstmTrainer

print("Training model...")
trainer = LstmTrainer(model, train_loader, val_loader, optimizer, True, criterion)
best_model = trainer.train(epochs=300, verbose=True, doEvaluate=True)
print("Model trained")

# Save the model
best_model.save("cache/lstm_model.pth")
