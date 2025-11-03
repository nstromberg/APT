"""Compare APT embedder/model behavior on CPU vs CUDA.

This script instruments the APT model (and falls back to a fresh APT when no checkpoint is
available) and prints expected vs true shapes as tensors are moved through the
embedding pipeline for modes: train, test and longitudinal.

Run:
    python scripts/compare_embedder_devices.py

The script will automatically skip CUDA if it's not available.
"""
from __future__ import annotations
import sys
import traceback
import numpy as np
import torch
import pprint

from apt.model import APT

try:
    from apt.model.embedder import APTEmbedder
    HAS_EMBEDDER = True
except Exception:
    APTEmbedder = None
    HAS_EMBEDDER = False


def build_fresh_model(device: str, small_config: dict | None = None) -> APT:
    # small config to keep compute light
    cfg = {
        "n_blocks": 2,
        "d_patch": 8,
        "d_model": 32,
        "d_ff": 64,
        "n_heads": 4,
        "dropout": 0.0,
        "activation": "gelu",
        "norm_eps": 1e-5,
        "classification": True,
    }
    if small_config:
        cfg.update(small_config)
    model = APT(**cfg)
    model.to(device)
    model.eval()
    return model


def log(msg, /, *args):
    print(msg.format(*args))


def instrument_and_run_on_device(device: str):
    log('\n=== Running on device: {} ===', device)
    device_torch = torch.device(device)

    # try to get an embedder; if not possible create fresh model
    model = None
    embedder = None
    used_embedder = False
    if HAS_EMBEDDER:
        try:
            embedder = APTEmbedder(device=device, verbose=True)
            used_embedder = True
            log('Using APTEmbedder (loaded checkpoint).')
            # expose model instance
            model = embedder.model
            model.to(device_torch)
            model.eval()
        except Exception as e:
            log('Could not instantiate APTEmbedder: {}', e)
            log('Falling back to a fresh APT with small config for testing.')
            model = build_fresh_model(device)
    else:
        log('APTEmbedder not importable; using fresh APT for testing.')
        model = build_fresh_model(device)

    # simple synthetic data
    seq_len = 12
    d_patch = model.d_patch
    # single sequence, single batch
    X = np.random.randn(1, seq_len, d_patch).astype(np.float32)
    y = np.random.randint(0, 2, size=(seq_len,), dtype=np.int64)

    # Move to torch tensors on the target device
    x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device_torch)

    results = {}

    # Mode: train
    try:
        n_train = seq_len // 2
        x_context = x_tensor[0, :n_train, :]
        x_query = x_tensor[0, n_train:, :]
        x_fold = torch.cat((x_context, x_query), dim=0).unsqueeze(0)  # (1, data_size, feat)
        y_context = torch.as_tensor(y[:n_train], dtype=torch.long, device=device_torch).unsqueeze(0)

        log('\n[train] x_fold.shape: {}', tuple(x_fold.shape))
        log('[train] y_context.shape: {}', tuple(y_context.shape))

        # log after embedding layers
        emb_x = model._emb_x(x_fold)
        log('[train] _emb_x output shape: {}', tuple(emb_x.shape))

        y_emb = model._emb_y(y_context.to(emb_x.dtype).unsqueeze(-1))
        log('[train] _emb_y output shape: {}', tuple(y_emb.shape))

        with torch.no_grad():
            out = model.get_query_embedding(x_fold, y_context)
        log('[train] get_query_embedding output shape: {} dtype={} device={}', tuple(out.shape), out.dtype, out.device)
        results['train'] = out.cpu().numpy()
    except Exception as e:
        results['train'] = ('error', str(e), traceback.format_exc())
        log('[train] ERROR: {}', e)

    # Mode: test
    try:
        # emulate stored train + test concatenation as embedder would do
        train_x = x_tensor[:, :n_train, :]
        test_x = x_tensor  # use full as test target for simplicity
        if train_x.shape[0] == 1 and test_x.shape[0] > 1:
            train_x = train_x.repeat(test_x.shape[0], 1, 1)

        batch_x = torch.cat([train_x.to(device_torch), test_x.to(device_torch)], dim=1)
        y_train = torch.as_tensor(y[:n_train], dtype=torch.long, device=device_torch).unsqueeze(0)
        if y_train.shape[0] == 1 and batch_x.shape[0] > 1:
            y_train = y_train.repeat(batch_x.shape[0], 1)

        log('\n[test] batch_x.shape: {}', tuple(batch_x.shape))
        log('[test] y_train.shape: {}', tuple(y_train.shape))

        emb_x = model._emb_x(batch_x)
        log('[test] _emb_x output shape: {}', tuple(emb_x.shape))

        with torch.no_grad():
            out = model.get_query_embedding(batch_x, y_train)
        log('[test] get_query_embedding output shape: {} dtype={} device={}', tuple(out.shape), out.dtype, out.device)
        results['test'] = out.cpu().numpy()
    except Exception as e:
        results['test'] = ('error', str(e), traceback.format_exc())
        log('[test] ERROR: {}', e)

    # Mode: longitudinal
    try:
        embeddings = []
        for i in range(1, seq_len):
            start = max(0, i - 4)  # window of 4
            context = x_tensor[0, start:i, :]
            query = x_tensor[0, i, :].unsqueeze(0)
            x_sample = torch.cat((context, query), dim=0).unsqueeze(0)
            y_dummy = torch.zeros((1, context.shape[0]), dtype=torch.long, device=device_torch)

            log('\n[longitudinal] i={} x_sample.shape={} y_dummy.shape={}', i, tuple(x_sample.shape), tuple(y_dummy.shape))

            emb_x = model._emb_x(x_sample)
            log('[longitudinal] _emb_x output shape: {}', tuple(emb_x.shape))

            with torch.no_grad():
                out = model.get_query_embedding(x_sample, y_dummy)
            log('[longitudinal] get_query_embedding out shape: {} dtype={} device={}', tuple(out.shape), out.dtype, out.device)
            embeddings.append(out.cpu().numpy())

        results['longitudinal'] = np.concatenate(embeddings, axis=0)
    except Exception as e:
        results['longitudinal'] = ('error', str(e), traceback.format_exc())
        log('[longitudinal] ERROR: {}', e)

    return results


def compare_results(cpu_res, cuda_res):
    print('\n\n=== Summary comparison CPU vs CUDA ===')
    for mode in ('train', 'test', 'longitudinal'):
        print('\nMode:', mode)
        a = cpu_res.get(mode)
        b = cuda_res.get(mode)
        if isinstance(a, tuple) and a[0] == 'error':
            print(' CPU ERROR:', a[1])
        else:
            print(' CPU shape:', None if a is None else getattr(a, 'shape', np.shape(a)))
        if b is None:
            print(' CUDA: not run')
        elif isinstance(b, tuple) and b[0] == 'error':
            print(' CUDA ERROR:', b[1])
        else:
            print(' CUDA shape:', getattr(b, 'shape', np.shape(b)))
        # if both arrays, compare numeric similarity
        if (not (isinstance(a, tuple) and a[0] == 'error')) and (not (isinstance(b, tuple) and b[0] == 'error')):
            try:
                a_np = np.asarray(a)
                b_np = np.asarray(b)
                if a_np.shape == b_np.shape:
                    diff = np.max(np.abs(a_np - b_np))
                    print(' max abs diff between CPU and CUDA outputs:', diff)
                else:
                    print(' shapes differ; cannot compute numeric diff')
            except Exception as e:
                print(' could not compare numerically:', e)


if __name__ == '__main__':
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    cpu_res = instrument_and_run_on_device('cpu')
    cuda_res = None
    if 'cuda' in devices:
        cuda_res = instrument_and_run_on_device('cuda')

    compare_results(cpu_res, cuda_res)

    print('\nDone. If shapes or devices differ unexpectedly, inspect printed logs above for the first mismatch.')
