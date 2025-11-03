import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FullAttention
from .feedforward import FeedForward
from .convolution import Conv1d


class TransformerBlock(nn.Module):
    """A Transformer block."""

    def __init__(self, d_model=512, n_heads=4, d_ff=2048,
                 dropout=0.1, activation="gelu",
                 norm_eps=1e-5, rope=False):
        """Initializes a new TransformerBlock instance.
        """
        super().__init__()
        self._attn = FullAttention(d_model, n_heads, dropout=dropout, rope=rope)
        self._ln_attn = nn.LayerNorm(d_model, eps=norm_eps)

        self._ff = FeedForward(d_ff, in_dim=d_model, out_dim=d_model,
            n_hid=1, activation=activation)
        self._ln_ff = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x, mask=None):
        # Diagnostic prints: if attention module has verbose turned on, print
        # shapes that influence q/k creation so we can trace unexpected dims.
        if getattr(self._attn, 'verbose', False):
            try:
                print(f"TransformerBlock.forward: input x.shape={tuple(x.shape)} mask.shape={None if mask is None else tuple(getattr(mask,'shape', ())) }")
            except Exception:
                pass

        x = x + self._attn(x, mask=mask)
        x = self._ln_attn(x)
        x = x + self._ff(x)
        x = self._ln_ff(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=128, d_model=512, n_heads=4, d_ff=2048,
                 dropout=0.0, activation="gelu", norm_eps=1e-5, rope=False):
        super().__init__()
        self.patch_size = patch_size

        if activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "tanh":
            act = nn.Tanh
        elif activation == "sigmoid":
            act = nn.Sigmoid
        else:
            act = activation

        self._patchify = nn.Sequential(
            Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size, padding=0),
            act(),
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        )

        self._emb = TransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, activation=activation,
            norm_eps=norm_eps, rope=rope
        )

        self._ln = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x):
        """
        x: (batch_size, data_size, feature_size)
        """
        try:
            b, l, f = x.size()
        except Exception:
            # fallback for unexpected shapes
            shape = tuple(x.shape)
            b = shape[0] if len(shape) > 0 else None
            l = shape[1] if len(shape) > 1 else None
            f = shape[2] if len(shape) > 2 else None

        # patchify operates on (batch*length, 1, features)
        x_patch = self._patchify(x.view(-1, 1, f))

        _, d, p = x_patch.size()

        # Treat patches as the sequence axis for the transformer: (batch*seq_len, p, d)
        x_patches_seq = x_patch.transpose(1, 2).contiguous()  # (batch*seq_len, p, d)

        # Diagnostic prints: show shapes that will affect attention's perceived
        # batch and sequence axes (this is where extra dims can appear).
        if getattr(self._emb._attn, 'verbose', False) or getattr(self, 'verbose', False):
            try:
                print(f"PatchEmbedding.forward: input.shape={(b,l,f)} patchified.shape={tuple(x_patch.shape)} patches_seq.shape={tuple(x_patches_seq.shape)}")
            except Exception:
                pass

        # Run transformer over patches for each original timestep (batch*seq_len as batch)
        out_patches = self._emb(x_patches_seq)  # (batch*seq_len, p, d_model)

        # Pool across the patch (sequence) dimension to get a single vector per timestep
        pooled = out_patches.sum(dim=1)  # (batch*seq_len, d_model)

        # reshape back to (batch, seq_len, d_model)
        x = pooled.view(b, l, -1)
        x = self._ln(x)
        return x


class MixtureBlock(nn.Module):
    """A Mixture block."""

    def __init__(self, d_model=512, n_heads=4, d_ff=2048,
                 dropout=0.1, activation="gelu", temperature=0.2):
        """Initializes a new MixtureBlock instance.
        """
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self._attn_logits = FeedForward(d_ff, in_dim=d_model, out_dim=d_model,
            n_hid=1, activation=activation)

        self._attn_gates = FeedForward(d_ff, in_dim=d_model, out_dim=d_model,
            n_hid=1, activation=activation)

        self.temperature = temperature

    def forward(self, hidden, split):
        """
        x: (batch_size, data_size, hidden_size)
        """
        b, l, _ = hidden.size()

        attn_gates = self._attn_gates(hidden)
        attn_gates = attn_gates.view(b, l, self.n_heads, -1).transpose(-3, -2)
        k_gates, q_gates = attn_gates[..., :split, :], attn_gates[..., split:, :]
        k_gates, q_gates = F.normalize(k_gates, dim=-1), F.normalize(q_gates, dim=-1)
        gates = torch.einsum(f"...ld, ...md -> ...lm", q_gates, k_gates)
        gates = torch.distributions.RelaxedBernoulli(
            self.temperature, logits=gates).rsample()

        attn_logits = self._attn_logits(hidden)
        attn_logits = attn_logits.view(b, l, self.n_heads, -1).transpose(-3, -2)
        k_logits, q_logits = attn_logits[..., :split, :], attn_logits[..., split:, :]
        logits = torch.einsum(f"...ld, ...md -> ...lm", q_logits, k_logits)
        probs = (logits * self.scale).softmax(dim=-1)
        probs = probs * gates
        probs = probs / probs.sum(-1, keepdim=True)

        probs = probs.mean(-3)
        return probs
