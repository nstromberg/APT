import torch
import torch.nn as nn

from .attention import FullAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """A Transformer block."""

    def __init__(self, d_model=512, n_heads=4, d_ff=2048,
                 dropout=0.1, activation="gelu", norm_eps=1e-5):
        """Initializes a new TransformerBlock instance.
        """
        super().__init__()
        self._attn = FullAttention(d_model, n_heads, dropout=dropout)
        self._ln_attn = nn.LayerNorm(d_model, eps=norm_eps)

        self._ff = FeedForward(d_ff, in_dim=d_model, out_dim=d_model,
            n_hid=1, activation=activation)
        self._ln_ff = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x, mask=None):
        x = x + self._attn(x, mask=mask)
        x = self._ln_attn(x)
        x = x + self._ff(x)
        x = self._ln_ff(x)

        return x
