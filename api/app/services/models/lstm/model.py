import torch
import torch.nn as nn
import pickle

class LSTM(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout_p: float = 0.1, *, VOCAB_SIZE: int, NUM_CLASSES: int, bidirectional: bool = True, with_attention: bool = True):
        super(LSTM, self).__init__()
        self.with_attention = with_attention
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            hidden_dim *= 2

        if with_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1)

        self.fc = nn.Linear(hidden_dim, NUM_CLASSES)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, _ = self.lstm(x)
        if self.with_attention:
            x, _ = self.attention(x, x, x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path + ".pt")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            model: LSTM = pickle.load(f)
            model.load_state_dict(torch.load(path + ".pt"))
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            return model
