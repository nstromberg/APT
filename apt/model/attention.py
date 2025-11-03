import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import set_device_config

# throttle SDPA failure warnings so logs don't get flooded
_sdpa_failure_warned = False


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
                 num_mem_kv=4, dropout=0.0, rope=False, flash=True, force_math=False):
        super().__init__(d_model, n_heads, d_in, num_mem_kv, dropout, rope)
        self.cpu_config, self.cuda_config = set_device_config(flash)
        self.flash = flash
        # Allow forcing the math (non-fused) attention path via env var
        # or constructor flag. Env var takes precedence.
        env_force = os.environ.get("APT_FORCE_MATH_ATTENTION")
        if env_force is not None:
            try:
                self.force_math = bool(int(env_force))
            except Exception:
                self.force_math = env_force.lower() in ("true", "1", "yes")
        else:
            self.force_math = bool(force_math)
        # Optionally try casting q/k/v to float16 for SDPA attempt on CUDA.
        # Controlled by env vars: APT_TRY_FP16_SDPA or APT_SDPA_FP16
        env_try_fp16 = os.environ.get("APT_TRY_FP16_SDPA") or os.environ.get("APT_SDPA_FP16")
        if env_try_fp16 is not None:
            try:
                self.try_fp16_sdpa = bool(int(env_try_fp16))
            except Exception:
                self.try_fp16_sdpa = str(env_try_fp16).lower() in ("true", "1", "yes")
        else:
            self.try_fp16_sdpa = False

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
        """Attend with either fused SDPA (when available) or the math fallback.

        This implementation supports forcing the math path via the
        `APT_FORCE_MATH_ATTENTION` env var or the `force_math` constructor flag.
        """
        # update kv with memory and caches
        k, v, kv_cache = self.update_kv(k, v, kv_cache=kv_cache)

        # pad mask if k has extra memory appended
        if mask is not None:
            mask = F.pad(mask, (k.shape[-2] - mask.shape[-1], 0), "constant", 1)

        def _math_attention(q_, k_, v_, mask_):
            attn = torch.einsum("...ld, ...md -> ...lm", q_, k_)
            attn = attn * self.scale
            if mask_ is not None:
                # ensure mask is a tensor on the correct device
                if not torch.is_tensor(mask_):
                    try:
                        mask_t = torch.as_tensor(mask_, device=attn.device)
                    except Exception:
                        mask_t = None
                else:
                    mask_t = mask_.to(attn.device)
                if mask_t is not None:
                    mask_bool = mask_t.bool()
                    try:
                        attn = attn.masked_fill(~mask_bool, float("-inf"))
                    except Exception:
                        attn = attn.masked_fill((mask_t == 0), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out_ = torch.einsum("...lm, ...me -> ...le", attn, v_)
            return out_

        # If user requested math-only, take that path immediately
        if getattr(self, 'force_math', False):
            out = _math_attention(q, k, v, mask)
            return out, kv_cache

        # Attempt fused SDPA/flash when enabled
        if self.flash:
            q, k, v = map(lambda t: t.contiguous(), (q, k, v))
            config = self.cuda_config if q.is_cuda else self.cpu_config

            if getattr(self, 'verbose', False):
                try:
                    print(f"FullAttention.attend: attempting SDPA; q.shape={tuple(q.shape)}, k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}, mask.shape={None if mask is None else tuple(mask.shape)}")
                except Exception:
                    pass

            def _reshape_for_sdpa(tensor):
                if tensor.ndim != 5:
                    return tensor, None
                shape = tuple(tensor.shape)
                try:
                    h_idx = next(i for i, s in enumerate(shape) if s == self.n_heads)
                    d_idx = next(i for i, s in enumerate(shape) if s == self.d_head)
                except StopIteration:
                    raise RuntimeError(f"Cannot identify head ({self.n_heads}) or head_dim ({self.d_head}) axes in tensor shape {shape}")
                others = [i for i in range(5) if i not in (h_idx, d_idx)]
                seq_idx = max(others, key=lambda i: shape[i])
                pre_batch = [i for i in others if i != seq_idx]
                if len(pre_batch) == 1:
                    perm = (pre_batch[0], h_idx, seq_idx, d_idx)
                    t_perm = tensor.permute(*perm).contiguous().unsqueeze(1)
                    a = shape[pre_batch[0]]
                    b = 1
                else:
                    perm = (pre_batch[0], pre_batch[1], h_idx, seq_idx, d_idx)
                    t_perm = tensor.permute(*perm).contiguous()
                    a = shape[pre_batch[0]]
                    b = shape[pre_batch[1]]
                a_times_b = a * b
                t_view = t_perm.view(a_times_b, self.n_heads, t_perm.shape[-2], self.d_head)
                t_out = t_view.transpose(1, 2).contiguous()
                metadata = {'orig_shape': shape, 'perm': perm, 'a': a, 'b': b, 'seq_dim_size': t_perm.shape[-2]}
                return t_out, metadata

            def _unreshape_from_sdpa(tensor, metadata):
                if metadata is None:
                    return tensor
                a = metadata['a']
                b = metadata['b']
                perm = metadata['perm']
                seq = metadata['seq_dim_size']
                tmp = tensor.transpose(1, 2).contiguous()
                tmp = tmp.view(a, b, self.n_heads, seq, self.d_head)
                inv_perm = [0] * len(perm)
                for i, p in enumerate(perm):
                    inv_perm[p] = i
                out = tmp.permute(*inv_perm).contiguous()
                return out

            try:
                # If tensors have 5 dims, reshape them for SDPA
                if q.ndim == 5 and k.ndim == 5 and v.ndim == 5 and q.shape[-1] == self.d_head:
                    q_s, q_meta = _reshape_for_sdpa(q)
                    k_s, k_meta = _reshape_for_sdpa(k)
                    v_s, v_meta = _reshape_for_sdpa(v)
                    q_meta = q_meta or {}

                    # Try to reshape mask to merged batch if possible
                    mask_s = None
                    if mask is not None:
                        try:
                            mask_t = mask
                            if not torch.is_tensor(mask_t):
                                mask_t = torch.as_tensor(mask_t, device=q.device)
                            mask_t = mask_t.bool()
                            a_q = q_meta.get('a', 1)
                            b_q = q_meta.get('b', 1)
                            if mask_t.ndim == 5 and mask_t.shape[0] == a_q and mask_t.shape[1] == b_q:
                                mask_s = mask_t.reshape(a_q * b_q, mask_t.shape[-2], mask_t.shape[-1])
                            elif mask_t.ndim in (3, 4) and mask_t.shape[0] == a_q * b_q:
                                mask_s = mask_t
                            else:
                                perm_q = q_meta.get('perm', None)
                                if perm_q is not None:
                                    try:
                                        mask_perm = mask_t.permute(*perm_q)
                                        mask_s = mask_perm.reshape(a_q * b_q, mask_perm.shape[-2], mask_perm.shape[-1])
                                    except Exception:
                                        mask_s = mask_t
                                else:
                                    mask_s = mask_t
                        except Exception:
                            mask_s = mask

                    # Don't attempt SDPA on CUDA when mask present
                    if mask_s is not None and q_s.is_cuda:
                        raise RuntimeError("Skipping SDPA because attn_mask present on CUDA")

                    # Optionally cast to float16 for the SDPA attempt on CUDA.
                    original_dtype = q_s.dtype
                    cast_back = False
                    q_h, k_h, v_h = q_s, k_s, v_s
                    if q_s.is_cuda and getattr(self, 'try_fp16_sdpa', False):
                        try:
                            q_h = q_s.half()
                            k_h = k_s.half()
                            v_h = v_s.half()
                            cast_back = True
                        except Exception:
                            q_h, k_h, v_h = q_s, k_s, v_s

                    try:
                        with torch.nn.attention.sdpa_kernel(config):
                            dropout_p = self.dropout if self.training else 0.0
                            out_s = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=None if mask_s is None else None, dropout_p=dropout_p, scale=self.scale)
                    except Exception as _sdpa_exc:
                        # Fall back to math attention on SDPA failure. Throttle the
                        # warning so we don't log the same failure repeatedly.
                        import warnings
                        global _sdpa_failure_warned
                        if not _sdpa_failure_warned:
                            warnings.warn(f"SDPA attempt failed ({_sdpa_exc}); falling back to math attention.", RuntimeWarning)
                            _sdpa_failure_warned = True
                        out = _math_attention(q, k, v, mask)
                        return out, kv_cache

                    if cast_back and out_s.dtype != original_dtype:
                        try:
                            out_s = out_s.float()
                        except Exception:
                            pass

                    out = _unreshape_from_sdpa(out_s, q_meta)
                else:
                    with torch.nn.attention.sdpa_kernel(config):
                        dropout_p = self.dropout if self.training else 0.0
                        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=self.scale)
            except Exception as exc:
                import warnings
                try:
                    qshape = tuple(q.shape)
                    kshape = tuple(k.shape)
                    vshape = tuple(v.shape)
                except Exception:
                    qshape = kshape = vshape = None
                warnings.warn(f"SDPA/flash attention path failed ({exc}); falling back to math attention. q.shape={qshape}, k.shape={kshape}, v.shape={vshape}", RuntimeWarning)
                out = _math_attention(q, k, v, mask)

            return out, kv_cache

        # Otherwise do math attention
        out = _math_attention(q, k, v, mask)
        return out, kv_cache
