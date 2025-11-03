import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import set_device_config


class RotaryPositionalEncoding(nn.Module):
  def __init__(self, dim: int, base: int = 10_000):
    super().__init__()
    self.dim = dim
    self.base = base
    self.cos_cached = None
    self.sin_cached = None

  def _neg_half(self, x: torch.Tensor):
    half_dim = self.dim // 2
    return torch.cat([-x[..., half_dim:], x[..., :half_dim]], dim=-1)

  def forward(self, x: torch.Tensor):
    *b, l, _, _ = x.shape
    n_b = len(b)
    if self.cos_cached is None or l > self.cos_cached.shape[n_b]:
        # build cache
        theta = 1. / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        seq_idx = torch.arange(l, device=x.device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[:, None, :][(None,) * n_b]
        self.sin_cached = idx_theta2.sin()[:, None, :][(None,) * n_b]

    neg_half_x = self._neg_half(x)
    x_rope = (x * self.cos_cached[..., :x.shape[n_b], :, :]) + (neg_half_x * self.sin_cached[..., :x.shape[n_b], :, :])
    return x_rope


class LinearAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_in=None,
                 num_mem_kv=4, dropout=0.0, rope=False):
        super().__init__()
        d_in = d_model if d_in is None else d_in

        self.n_heads = n_heads
        self.dropout = dropout
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.to_qkv = nn.Linear(d_in, d_model * 3, bias=False)
        self.mem_kv = nn.Parameter(torch.randn(2, n_heads, num_mem_kv, self.d_head))
        self.rope = RotaryPositionalEncoding(self.d_head) if rope else None

    def forward(self, x, mask=None, kv_cache=False):
        """
        x: (..., sequence_length, feature_size)
        """
        if mask is not None:
            mask = self.prepare_mask(mask, x_ndim=x.ndim)
        *b, l, _ = x.shape

        if getattr(self, 'verbose', False):
            try:
                print(f"LinearAttention.forward: x.shape={tuple(x.shape)} expanded batch_dims={tuple(b)} seq_len={l} n_heads={self.n_heads} d_head={self.d_head}")
            except Exception:
                pass

        qkv = self.to_qkv(x).view(*b, l, self.n_heads, -1, 3).transpose(-4, -3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if getattr(self, 'verbose', False):
            try:
                print(f"LinearAttention.forward: q.shape={tuple(q.shape)} k.shape={tuple(k.shape)} v.shape={tuple(v.shape)}")
            except Exception:
                pass
        if self.rope is not None:
            qk = self.rope(torch.stack((q, k), dim=0))
            q, k = qk[0], qk[1]

        out, kv_cache = self.attend(q, k, v, mask=mask, kv_cache=kv_cache)

        out = out.transpose(-3, -2).contiguous().view(*b, l, -1)
        if kv_cache is not None:
            return out, kv_cache
        return out

    def prepare_mask(self, mask, x_ndim=3):
        """
        mask: (..., sequence_length)
            *mask is node-level in linear attention
        """
        # source-to-target
        if mask.ndim < x_ndim - 1:
            mask = mask[(None,)*(x_ndim-1-mask.ndim)]
        return mask[..., None, :, None]

    def update_kv(self, k, v, kv_cache=False):
        """
        kv_cache: True, False, or (k_cache, v_cache)
        """
        if kv_cache is not False:
            if kv_cache is not True:
                k_cache, v_cache = kv_cache
                k, v = map(functools.partial(torch.cat, dim=-2), ((k_cache, k), (v_cache, v)))
            kv_cache = (k, v)
        else:
            kv_cache = None
        mk, mv = map(lambda t: t.repeat(*k.shape[:-t.ndim], *[1]*t.ndim), self.mem_kv)
        k, v = map(functools.partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        return k, v, kv_cache

    def attend(self, q, k, v, mask=None, kv_cache=False):
        if mask is not None:
            k = (k * mask).masked_fill(mask == 0, float('-inf'))

        k, v, kv_cache = self.update_kv(k, v, kv_cache=kv_cache)

        q = q * self.scale
        q = q.softmax(dim=-1)
        q = F.dropout(q, p=self.dropout, training=self.training)
        k = k.softmax(dim=-2)
        context = torch.einsum('...nd, ...ne -> ...de', k, v)
        out = torch.einsum('...ld, ...de -> ...le', q, context)
        if mask is not None:
            out = out * mask

        return out, kv_cache


class FullAttention(LinearAttention):
    def __init__(self, d_model=128, n_heads=4, d_in=None,
                 num_mem_kv=4, dropout=0.0, rope=False, flash=True):
        super().__init__(d_model, n_heads, d_in, num_mem_kv, dropout, rope)
        self.cpu_config, self.cuda_config = set_device_config(flash)
        self.flash = flash

    def prepare_mask(self, mask, x_ndim=3):
        """
        mask: (..., sequence_length, sequence_length)
            *mask is edge-level in full attention
        """
        # target-to-source
        if mask.ndim < x_ndim:
            mask = mask[(None,)*(x_ndim-mask.ndim)]
        return mask[..., None, :, :]

    def attend(self, q, k, v, mask=None, kv_cache=False):
        k, v, kv_cache = self.update_kv(k, v, kv_cache=kv_cache)

        if mask is not None:
            mask = F.pad(mask, (k.shape[-2] - mask.shape[-1], 0), "constant", 1)

        if self.flash:
            q, k, v = map(lambda t: t.contiguous(), (q, k, v))
            # Check if there is a compatible device for flash attention
            config = self.cuda_config if q.is_cuda else self.cpu_config
            # print(q.shape, k.shape, v.shape)

            # Try SDPA/flash attention; if q/k/v are 5-D (extra leading dims)
            # reshape to 4-D (batch, seq_len, num_heads, head_dim) before
            # calling the fused kernel, then reshape the output back.
            if getattr(self, 'verbose', False):
                print(f"FullAttention.attend: attempting SDPA; q.shape={tuple(q.shape)}, k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}, mask.shape={None if mask is None else tuple(mask.shape)}")

            def _reshape_for_sdpa(tensor):
                """
                Robustly reshape a 5-D tensor (with two leading batch-like dims)
                into the 4-D shape expected by the fused SDPA kernel:

                    input (a, b, h, L, d) -> output (a*b, L, h, d)

                This function is defensive: it locates the head and head-dim
                axes by matching sizes (self.n_heads and self.d_head),
                permutes to a canonical ordering, then merges leading batch
                dims. It returns (reshaped_tensor, metadata) where metadata
                contains information required to un-reshape the output.
                """
                if tensor.ndim != 5:
                    return tensor, None

                shape = tuple(tensor.shape)
                # find axis indices for head count and head dim
                try:
                    h_idx = next(i for i, s in enumerate(shape) if s == self.n_heads)
                    d_idx = next(i for i, s in enumerate(shape) if s == self.d_head)
                except StopIteration:
                    # Could not identify expected axes; bail out
                    raise RuntimeError(f"Cannot identify head ({self.n_heads}) or head_dim ({self.d_head}) axes in tensor shape {shape}")

                # Remaining axes are pre-batch dims and the sequence axis.
                others = [i for i in range(5) if i not in (h_idx, d_idx)]
                # Heuristic: choose the sequence axis as the largest remaining dim
                seq_idx = max(others, key=lambda i: shape[i])
                pre_batch = [i for i in others if i != seq_idx]

                # Build permutation to (pre_batch[0], pre_batch[1], h_idx, seq_idx, d_idx)
                if len(pre_batch) == 1:
                    perm = (pre_batch[0], h_idx, seq_idx, d_idx)
                    # inject a singleton so we always have two pre-batch dims when viewing
                    t_perm = tensor.permute(*perm).contiguous().unsqueeze(1)
                    a = shape[pre_batch[0]]
                    b = 1
                else:
                    perm = (pre_batch[0], pre_batch[1], h_idx, seq_idx, d_idx)
                    t_perm = tensor.permute(*perm).contiguous()
                    a = shape[pre_batch[0]]
                    b = shape[pre_batch[1]]

                # now t_perm has shape (a, b, h, L, d)
                # merge leading batch dims and reorder to (a*b, L, h, d)
                a_times_b = a * b
                # view/reshape then transpose heads and seq dims
                t_view = t_perm.view(a_times_b, self.n_heads, t_perm.shape[-2], self.d_head)
                t_out = t_view.transpose(1, 2).contiguous()  # (a*b, L, h, d)

                metadata = {
                    'orig_shape': shape,
                    'perm': perm,
                    'a': a,
                    'b': b,
                    'seq_dim_size': t_perm.shape[-2],
                }
                return t_out, metadata

            def _unreshape_from_sdpa(tensor, metadata):
                """Reverse the operation performed by _reshape_for_sdpa."""
                if metadata is None:
                    return tensor
                a = metadata['a']
                b = metadata['b']
                perm = metadata['perm']
                seq = metadata['seq_dim_size']

                # tensor: (a*b, L, h, d)
                tmp = tensor.transpose(1, 2).contiguous()  # (a*b, h, L, d)
                # view back to (a, b, h, L, d)
                tmp = tmp.view(a, b, self.n_heads, seq, self.d_head)

                # Compute inverse permute
                inv_perm = [0] * len(perm)
                for i, p in enumerate(perm):
                    inv_perm[p] = i

                # tmp currently ordered as (pre0, pre1, h, L, d) matching perm; invert.
                out = tmp.permute(*inv_perm).contiguous()
                return out
            try:
                reshaped = False
                if q.ndim == 5 and k.ndim == 5 and v.ndim == 5 and q.shape[-1] == self.d_head:
                    q_s, q_orig = _reshape_for_sdpa(q)
                    k_s, k_orig = _reshape_for_sdpa(k)
                    v_s, v_orig = _reshape_for_sdpa(v)

                    # adjust mask for merged batch dim when possible
                    mask_s = None
                    if mask is not None:
                        try:
                            mask_t = mask
                            if not torch.is_tensor(mask_t):
                                mask_t = torch.as_tensor(mask_t, device=q.device)
                            mask_t = mask_t.bool()

                            # mask_t may have shape (..., seq, seq) or include pre-batch dims
                            q_meta = q_orig or {}
                            a_q = q_meta.get('a', 1)
                            b_q = q_meta.get('b', 1)
                            if mask_t.ndim == 5 and mask_t.shape[0] == a_q and mask_t.shape[1] == b_q:
                                mask_s = mask_t.reshape(a_q * b_q, mask_t.shape[-2], mask_t.shape[-1])
                            elif mask_t.ndim == 4 and mask_t.shape[0] == a_q * b_q:
                                mask_s = mask_t
                            elif mask_t.ndim == 3 and mask_t.shape[0] == a_q * b_q:
                                mask_s = mask_t
                            else:
                                # Try permuting mask according to the permutation we used for q/k/v
                                try:
                                    perm_q = q_meta.get('perm', None)
                                    if perm_q is not None:
                                        mask_perm = mask_t.permute(*perm_q)
                                        mask_s = mask_perm.reshape(a_q * b_q, mask_perm.shape[-2], mask_perm.shape[-1])
                                    else:
                                        mask_s = mask_t
                                except Exception:
                                    # Give up and pass the original mask through; SDPA may still reject it
                                    mask_s = mask_t
                        except Exception:
                            mask_s = mask

                    # Don't attempt flash/SDPA if an attn mask is present on CUDA
                    # because many fused kernels don't support non-null masks.
                    can_try_sdpa = True
                    if mask_s is not None and q_s.is_cuda:
                        can_try_sdpa = False

                    if can_try_sdpa:
                        # If on CUDA and tensors are float32, try casting to float16
                        # for the fused kernel (it expects half/bfloat16). Cast back
                        # the output afterwards to preserve the model dtype.
                        original_dtype = q_s.dtype
                        cast_back = False
                        if q_s.is_cuda and q_s.dtype == torch.float32:
                            try:
                                q_s_h = q_s.half()
                                k_s_h = k_s.half()
                                v_s_h = v_s.half()
                                cast_back = True
                            except Exception:
                                q_s_h = q_s
                                k_s_h = k_s
                                v_s_h = v_s
                        else:
                            q_s_h = q_s
                            k_s_h = k_s
                            v_s_h = v_s

                        with torch.nn.attention.sdpa_kernel(config):
                            dropout_p = self.dropout if self.training else 0.0
                            out_s = F.scaled_dot_product_attention(q_s_h, k_s_h, v_s_h,
                                attn_mask=None if mask_s is None else None, dropout_p=dropout_p, scale=self.scale)

                        if cast_back and out_s.dtype != original_dtype:
                            try:
                                out_s = out_s.float()
                            except Exception:
                                pass
                    else:
                        # Not safe to run SDPA (mask present on CUDA), raise to trigger fallback
                        raise RuntimeError("Skipping SDPA because attn_mask present on CUDA")

                    out = _unreshape_from_sdpa(out_s, q_orig)
                    reshaped = True
                else:
                    with torch.nn.attention.sdpa_kernel(config):
                        dropout_p = self.dropout if self.training else 0.0
                        out = F.scaled_dot_product_attention(q, k, v,
                            attn_mask=mask, dropout_p=dropout_p, scale=self.scale)
            except Exception as exc:
                # Some CUDA environments may raise different errors or warnings when
                # no compatible SDPA kernel exists. Fall back to the safe math-based
                # attention implementation and emit a warning with shapes to aid debugging.
                import warnings
                try:
                    qshape = tuple(q.shape)
                    kshape = tuple(k.shape)
                    vshape = tuple(v.shape)
                except Exception:
                    qshape = kshape = vshape = None
                warnings.warn(
                    f"SDPA/flash attention path failed ({exc}); falling back to math attention. q.shape={qshape}, k.shape={kshape}, v.shape={vshape}",
                    RuntimeWarning,
                )
                attn = torch.einsum(f"...ld, ...md -> ...lm", q, k)
                attn = attn * self.scale
                # ensure mask is a tensor on the correct device and dtype
                if mask is not None:
                    if not torch.is_tensor(mask):
                        try:
                            mask_t = torch.as_tensor(mask, device=attn.device)
                        except Exception:
                            mask_t = None
                    else:
                        mask_t = mask.to(attn.device)
                    if mask_t is not None:
                        # convert to boolean mask and broadcast if needed
                        mask_bool = mask_t.bool()
                        try:
                            attn = attn.masked_fill(~mask_bool, float('-inf'))
                        except Exception:
                            # fallback: try comparison-style mask
                            attn = attn.masked_fill((mask_t == 0), float('-inf'))
                attn = attn.softmax(dim=-1)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                out = torch.einsum(f"...lm, ...me -> ...le", attn, v)
        else:
            attn = torch.einsum(f"...ld, ...md -> ...lm", q, k)
            attn = attn * self.scale
            attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.einsum(f"...lm, ...me -> ...le", attn, v)

        return out, kv_cache
