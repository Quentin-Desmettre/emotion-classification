
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
from .lstm import LSTM
from torch import nn

class LstmTrainer:
    def __init__(self, model: LSTM, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, doEvaluate: bool, criterion: nn.Module):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.doEvaluate = doEvaluate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion

    def train(self, epochs: int, *, verbose: bool = True, doEvaluate: bool = True):
        best_model = None
        best_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in tqdm(self.train_loader) if verbose else self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            if not doEvaluate:
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}") if verbose else None
                continue

            self.model.eval()
            val_loss = 0
            val_preds, val_labels = [], []
            total_preds = 0
            total_correct = 0
            with torch.no_grad():
                print ("Validating the model") if verbose else None
                for batch in self.val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())

                    total_preds += len(preds)
                    total_correct += (preds == labels.cpu().numpy()).sum()

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.model)

            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {total_correct / total_preds:.4f}") if verbose else None
        return best_model
