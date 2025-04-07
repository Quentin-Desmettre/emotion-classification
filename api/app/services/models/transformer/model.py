import torch
import torch.nn as nn
import pickle
import math


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        num_classes,
        max_len,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(
            self._get_positional_encoding(max_len, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, num_classes)

    def _get_positional_encoding(self, max_len, embed_dim):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x) + self.positional_encoding[:, : x.size(1), :]
        if attention_mask is not None:
            # Convert attention_mask to the expected format for `nn.TransformerEncoder`
            attention_mask = (
                attention_mask == 0
            )  # Mask padded tokens (True for padding)
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        x = x.mean(dim=1)  # Global pooling
        logits = self.fc(x)
        return logits

    def save(self, path: str):
        torch.save(self.state_dict(), path + ".pt")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            model: TransformerModel = pickle.load(f)
            model.load_state_dict(torch.load(path + ".pt"))
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            return model

    def to(self, device):
        super(TransformerModel, self).to(device)
        self.embedding.to(device)
        self.positional_encoding.to(device)
        self.transformer_encoder.to(device)
        self.fc.to(device)
        return self
