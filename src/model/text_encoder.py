"""Lightweight BiLSTM text encoder with FiLM generator heads."""

import torch
import torch.nn as nn

from src.data.prompt_pool import VOCAB_SIZE, MAX_PROMPT_LEN
from src.model.film import FiLMGenerator


class TextEncoder(nn.Module):
    """Lightweight Bidirectional LSTM text encoder.
    
    Encodes character-level tokenized prompts into a fixed-length
    conditioning vector, then generates FiLM parameters for each
    encoder/decoder stage.
    
    Architecture:
        Embedding(vocab_size, 64) -> BiLSTM(128) -> GlobalAvgPool -> FC(256)
        -> FiLM generators for each stage
    """

    def __init__(self, vocab_size=None, embed_dim=64, hidden_size=128,
                 num_layers=1, bidirectional=True, stage_channels=None):
        super().__init__()

        if vocab_size is None:
            vocab_size = VOCAB_SIZE
        if stage_channels is None:
            stage_channels = [64, 128, 256, 512, 256, 128, 64, 32]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        cond_dim = 256
        self.fc = nn.Linear(lstm_out_dim, cond_dim)

        # FiLM generators for each stage
        self.film_generators = nn.ModuleList([
            FiLMGenerator(cond_dim, ch) for ch in stage_channels
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        if self.lstm.bidirectional:
            nn.init.orthogonal_(self.lstm.weight_ih_l0_reverse)
            nn.init.orthogonal_(self.lstm.weight_hh_l0_reverse)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, token_ids):
        """
        Args:
            token_ids: (B, seq_len) int64 token IDs
        Returns:
            film_params: list of (gamma, beta) tuples for each stage
        """
        x = self.embedding(token_ids)  # (B, seq_len, embed_dim)
        output, _ = self.lstm(x)       # (B, seq_len, lstm_out_dim)

        # Global average pool over sequence
        # Mask out padding (token_ids == 0)
        mask = (token_ids != 0).unsqueeze(-1).float()  # (B, seq_len, 1)
        lengths = mask.sum(dim=1).clamp(min=1)          # (B, 1)
        pooled = (output * mask).sum(dim=1) / lengths    # (B, lstm_out_dim)

        cond = torch.relu(self.fc(pooled))  # (B, cond_dim)

        film_params = [gen(cond) for gen in self.film_generators]
        return film_params
