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
import os
import random
import argparse

# Prefer a relative import when the script is executed as a module (python -m),
# otherwise fall back to inserting the repo root on sys.path so we import the
# local `apt` package instead of any installed distribution.
import pathlib

try:
    # when run via `python -m scripts.compare_embedder_devices` __package__ is set
    if __package__:
        from .apt.model import APT  # type: ignore
        from .apt.model.embedder import APTEmbedder  # type: ignore
        HAS_EMBEDDER = True
    else:
        raise ImportError
except Exception:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

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

    # Ensure deterministic runs and consistent attention path for parity testing.
    # Force the math (non-fused) attention path for both CPU and CUDA so we
    # compare identical algorithms. Also set deterministic flags and seeds.
    os.environ.setdefault('APT_FORCE_MATH_ATTENTION', '1')
    # Deterministic behavior for reproducibility
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # try to get an embedder; if not possible create fresh model
    model = None
    embedder = None
    used_embedder = False
    if HAS_EMBEDDER and APTEmbedder is not None:
        try:
            embedder = APTEmbedder(device=device, verbose=True)
            used_embedder = True
            log('Using APTEmbedder (loaded checkpoint).')
            # expose model instance
            model = embedder.model
            model.to(device_torch)
            model.eval()
            # report attention math forcing for debugging
            fm_total = 0
            fm_true = 0
            for m in model.modules():
                if hasattr(m, 'force_math'):
                    fm_total += 1
                    if getattr(m, 'force_math'):
                        fm_true += 1
            log('Model attention modules with force_math: {}/{}', fm_true, fm_total)
        except Exception as e:
            log('Could not instantiate APTEmbedder: {}', e)
            log('Falling back to a fresh APT with small config for testing.')
            model = build_fresh_model(device)
    else:
        log('APTEmbedder not importable; using fresh APT for testing.')
        model = build_fresh_model(device)

    # simple synthetic data
    seq_len = 12
    n_train = seq_len // 2
    d_patch = model.d_patch
    # single sequence, single batch
    X = np.random.randn(1, seq_len, d_patch).astype(np.float32)
    y = np.random.randint(0, 2, size=(seq_len,), dtype=np.int64)

    # Move to torch tensors on the target device
    x_tensor = torch.as_tensor(X, dtype=torch.float32, device=device_torch)

    # If we successfully created an APTEmbedder, fit it on the synthetic
    # training data so we can call its public `transform` API below.
    if used_embedder and embedder is not None:
        try:
            # embedder.fit expects X in array-like shape (n_sequences, max_len, n_features)
            embedder.fit(X, y)
            xtrain_shape = None
            ytrain_shape = None
            if getattr(embedder, 'x_train', None) is not None:
                xtrain_shape = getattr(embedder, 'x_train').shape
            if getattr(embedder, 'y_train', None) is not None:
                ytrain_shape = getattr(embedder, 'y_train').shape
            log('Fitted APTEmbedder: x_train.shape={} y_train.shape={}', xtrain_shape, ytrain_shape)
        except Exception as e:
            log('Could not fit APTEmbedder on synthetic data: {}', e)
            # fallback to using model internals
            used_embedder = False

    results = {}

    # Mode: train
    try:
        if used_embedder and embedder is not None:
            # Use the embedder public API for train embeddings
            try:
                out_np = embedder.transform(X, mode='train', k_folds=2)
                log('[train] embedder.transform output shape: {}', getattr(out_np, 'shape', np.shape(out_np)))
                results['train'] = out_np
            except Exception as e:
                results['train'] = ('error', str(e), traceback.format_exc())
                log('[train] embedder.transform ERROR: {}', e)
        else:
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


def _extract_init_args_from_model(model: APT) -> dict:
    """Reconstruct init args from an instantiated APT model instance."""
    return {
        "n_blocks": getattr(model, "n_blocks"),
        "d_patch": getattr(model, "d_patch"),
        "d_model": getattr(model, "d_model"),
        "d_ff": getattr(model, "d_ff"),
        "n_heads": getattr(model, "n_heads"),
        "dropout": getattr(model, "dropout"),
        "activation": getattr(model, "activation"),
        "norm_eps": getattr(model, "norm_eps"),
        "classification": getattr(model, "classification"),
    }


def _tensor_from_out(out):
    """Normalize module forward output to a single Tensor for comparison.

    If module returns a tuple like (out, kv_cache), pick the primary tensor.
    """
    if isinstance(out, torch.Tensor):
        return out.detach()
    if isinstance(out, (list, tuple)) and len(out) > 0:
        for x in out:
            if isinstance(x, torch.Tensor):
                return x.detach()
    # fallback: try to convert whatever it is to a tensor
    try:
        return torch.as_tensor(out).detach()
    except Exception:
        return None


def per_layer_compare(tol: float = 1e-4, save_divergent: bool = False):
    """Run the same inputs through CPU and CUDA models and compare intermediate activations.

    Registers forward hooks on PatchEmbedding, TransformerBlock, FullAttention and
    LinearAttention modules and reports per-module max abs diffs.
    """
    if not torch.cuda.is_available():
        print('CUDA not available; cannot run per-layer comparator.')
        return

    # Build or obtain a reference model to extract state_dict and init args
    try:
        if HAS_EMBEDDER and APTEmbedder is not None:
            ref_embedder = APTEmbedder(device='cpu', verbose=False)
            ref_model = ref_embedder.model
        else:
            ref_model = build_fresh_model('cpu')
    except Exception:
        ref_model = build_fresh_model('cpu')

    state = ref_model.state_dict()
    init_args = _extract_init_args_from_model(ref_model)

    # Build CPU and CUDA models from the same state
    cpu_model = APT(**init_args)
    cpu_model.load_state_dict(state)
    cpu_model.to('cpu')
    cpu_model.eval()

    cuda_model = APT(**init_args)
    cuda_model.load_state_dict(state)
    cuda_model.to('cuda')
    cuda_model.eval()

    # Create simple synthetic inputs matching other runs
    seq_len = 12
    n_train = seq_len // 2
    d_patch = cpu_model.d_patch
    X = np.random.randn(1, seq_len, d_patch).astype(np.float32)
    y = np.random.randint(0, 2, size=(seq_len,), dtype=np.int64)

    x_tensor_cpu = torch.as_tensor(X, dtype=torch.float32, device='cpu')
    x_tensor_cuda = torch.as_tensor(X, dtype=torch.float32, device='cuda')

    # Build batch inputs used in instrumented runs
    train_x_cpu = x_tensor_cpu[:, :n_train, :]
    test_x_cpu = x_tensor_cpu
    batch_x_cpu = torch.cat([train_x_cpu, test_x_cpu], dim=1)
    y_train_cpu = torch.as_tensor(y[:n_train], dtype=torch.long, device='cpu').unsqueeze(0)

    train_x_cuda = x_tensor_cuda[:, :n_train, :]
    test_x_cuda = x_tensor_cuda
    batch_x_cuda = torch.cat([train_x_cuda, test_x_cuda], dim=1)
    y_train_cuda = torch.as_tensor(y[:n_train], dtype=torch.long, device='cuda').unsqueeze(0)

    # Select modules to hook (by class name)
    target_names = set(['PatchEmbedding', 'TransformerBlock', 'FullAttention', 'LinearAttention', 'MixtureBlock'])

    cpu_acts = {}
    cuda_acts = {}

    hooks = []

    def register_hooks(model, storage, device_name):
        for name, mod in model.named_modules():
            if mod.__class__.__name__ in target_names:
                # create local var to capture name
                def make_hook(n):
                    def hook(module, inp, out):
                        t = _tensor_from_out(out)
                        if t is None:
                            storage[n] = None
                            return
                        # move to cpu for uniform comparison
                        try:
                            storage[n] = t.detach().cpu().clone()
                        except Exception:
                            try:
                                storage[n] = t.detach().cpu()
                            except Exception:
                                storage[n] = None
                    return hook
                h = mod.register_forward_hook(make_hook(name))
                hooks.append(h)

    # register hooks
    register_hooks(cpu_model, cpu_acts, 'cpu')
    register_hooks(cuda_model, cuda_acts, 'cuda')

    # Run forward passes
    with torch.no_grad():
        try:
            _ = cpu_model.get_query_embedding(batch_x_cpu, y_train_cpu)
        except Exception as e:
            print('CPU model forward failed during per-layer run:', e)
        try:
            _ = cuda_model.get_query_embedding(batch_x_cuda, y_train_cuda)
        except Exception as e:
            print('CUDA model forward failed during per-layer run:', e)

    # remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # Compare activations in the order encountered in cpu model
    print('\n=== Per-layer comparison (max abs diff) ===')
    divergences = {}
    for name in cpu_acts.keys():
        a = cpu_acts.get(name)
        b = cuda_acts.get(name)
        if a is None or b is None:
            print(f"{name}: one side missing activation (cpu={'yes' if a is not None else 'no'}, cuda={'yes' if b is not None else 'no'})")
            continue
        if a.shape != b.cpu().shape:
            print(f"{name}: shape mismatch cpu={tuple(a.shape)} cuda={tuple(b.cpu().shape)}")
            divergences[name] = ('shape', tuple(a.shape), tuple(b.cpu().shape))
            continue
        diff = torch.max(torch.abs(a - b.cpu())).item()
        print(f"{name}: max_abs_diff={diff}")
        if diff > tol:
            divergences[name] = ('value', diff)
            if save_divergent:
                fname = f"divergence_{name.replace('/', '_')}.pt"
                torch.save({'cpu': a, 'cuda': b.cpu()}, fname)
                print(f" Saved divergent tensors to {fname}")

    if divergences:
        print('\nFirst divergences found:')
        for k, v in divergences.items():
            print(k, v)
    else:
        print('\nNo per-layer divergences above tolerance found.')




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

    # Optionally run the per-layer comparator when requested via environment.
    # Set APT_PER_LAYER=1 to enable. Optional extras:
    #   APT_PER_LAYER_TOL to set tolerance (float), default 1e-4
    #   APT_PER_LAYER_SAVE=1 to save divergent tensors to disk
    if os.environ.get('APT_PER_LAYER') in ('1', 'true', 'True'):
        try:
            tol = float(os.environ.get('APT_PER_LAYER_TOL', '1e-4'))
        except Exception:
            tol = 1e-4
        save_div = os.environ.get('APT_PER_LAYER_SAVE') in ('1', 'true', 'True')
        print('\nRunning per-layer comparator (tol={} save_divergent={})'.format(tol, save_div))
        per_layer_compare(tol=tol, save_divergent=save_div)

    print('\nDone. If shapes or devices differ unexpectedly, inspect printed logs above for the first mismatch.')
