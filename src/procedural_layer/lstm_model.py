"""Bidirectional LSTM multi-label sequence classifier for procedural tags."""

import torch
from torch import nn


class ProceduralLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        _, (hidden, _) = self.lstm(embedded)
        # concat final forward + backward hidden states of the last layer
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.classifier(self.dropout(final_hidden))
