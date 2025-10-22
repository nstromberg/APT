from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore
import torch  # type: ignore
import numpy as np
from apt.model import APT
from sklearn.model_selection import StratifiedKFold, KFold  # type: ignore
import pathlib
import os
import warnings

class APTEmbedder(TransformerMixin, BaseEstimator):
    def __init__(self, device="cpu", model_name="model_epoch=200_classification_2025.01.13_21:18:53.pt",
                 base_path=pathlib.Path(__file__).parent.parent.resolve(), model_dir="checkpoints",
                 url="https://osf.io/download/684c9eb0fdbd7bc7fab689be/"):
        """
        Initialize the APTEmbedder.

        Parameters:
        model_name : str
            Name of the pre-trained model file.
        device : str, optional
            Device to run the model on. If None, it uses "cuda" if available, otherwise "cpu".
        base_path : str, optional
            Base path for the model directory.
        model_dir : str, optional
            Directory where the model is stored.
        url : str, optional
            URL to download the model if not found locally.
        """
        self.x_train = None
        self.y_train = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(base_path, model_dir, model_name)
        self.model_path = model_path
        if not pathlib.Path(model_path).is_file():
            print(f"Model not found at {model_path}. Downloading from {url}...")
            import requests  # type: ignore
            r = requests.get(url, allow_redirects=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, 'wb').write(r.content)

        state_dict, init_args = torch.load(self.model_path, map_location='cpu')
        self.model = APT(**init_args)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'APTEmbedder':
        """
        Fit the model to the training data.

        Parameters:
        X : array-like
            Training data.
        y : array-like, optional
            Target values. If None, a zero array is created.

        Returns:
        self : object
            Returns the instance itself.
        """
        X_t = torch.as_tensor(X, dtype=torch.float32)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(0)
        self.x_train = X_t.to(self.device)

        if y is None:
            y = np.zeros((self.x_train.shape[1],), dtype=np.float32)
        y_t = torch.as_tensor(y)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(0)
        if y_t.shape[-1] != self.x_train.shape[1]:
            raise ValueError("Length of y must match number of timesteps in X.")
        self.y_train = y_t.to(self.device)
        return self

    def transform(self, X: np.ndarray, mode: str = "train", k_folds: int = 5, k: int = 5, fixed_window: bool = True) -> np.ndarray:
        """
        Transform the input data based on the specified mode.

        Parameters:
        X : array-like
            Input data to transform.
        mode : str, optional
            Mode of transformation: "train", "test", or "longitudinal".
        k_folds : int, optional
            Number of folds for training mode.
        k: int, optional
            Size of (initial) window for longitudinal embeddings.
        fixed_window : bool, optional
            If True, keep only the previous k points; if False, keep all points before the current one.

        Returns:
        np.ndarray
            Transformed data.
        """
        if (self.x_train is None or self.y_train is None) and mode=='test':
            raise ValueError("Call fit with training data before transform.")

        X = torch.as_tensor(X).to(self.device)  # (n_sequences, max_len, n_features)
        if X.ndim == 2:
            X = X.unsqueeze(0)

        results = []
        max_steps = 0  # Track the maximum steps for padding
        for seq in X:
            # Remove nan padding (assume nan in all features means padding)
            mask = ~torch.isnan(seq).all(dim=-1)
            seq_clean = seq[mask]
            if seq_clean.shape[0] == 0:
                # If sequence is all padding, skip or fill with nan
                results.append(np.full((1, self.model.d_model), np.nan))
                continue
            if mode == "train":
                transformed = self._transform_train(seq_clean, k_folds=k_folds)
            elif mode == "test":
                transformed = self._transform_test(seq_clean)
            elif mode == "longitudinal":
                transformed = self._transform_longitudinal(seq_clean, k=k, fixed_window=fixed_window)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            results.append(transformed)
            max_steps = max(max_steps, transformed.shape[0])  # Update max_steps

        # Pad results to ensure they all have the same shape
        padded_results = np.full((len(results), max_steps, self.model.d_model), np.nan)
        for i, result in enumerate(results):
            padded_results[i, :result.shape[0], :] = result

        # Emit a lightweight diagnostic if embeddings collapse to a single value
        if np.nanstd(padded_results) < 1e-8:
            warnings.warn(
                "Transformed embeddings have near-zero variance; check context construction or mode-specific inputs.",
                RuntimeWarning,
            )

        return padded_results

    def _transform_train(self, x: torch.Tensor, y: torch.Tensor = None, k_folds: int = 5) -> np.ndarray:
        """
        Transform the training data using stratified k-folds.

        Parameters:
        x : tensor
            Input training data.
        y : tensor, optional
            Target values.
        k_folds : int, optional
            Number of folds.

        Returns:
        np.ndarray
            Embeddings for the training data.
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)

        seq_len = x.shape[0]
        if seq_len < 2:
            return np.zeros((0, self.model.d_model), dtype=np.float32)

        if k_folds > seq_len:
            k_folds = seq_len

        if y is None:
            y = np.zeros(seq_len, dtype=np.int64)
        y_np = np.asarray(y)
        if y_np.shape[0] != seq_len:
            raise ValueError("Length of labels must match sequence length for train transform.")

        skf = StratifiedKFold(n_splits=k_folds)
        try:
            split_iter = list(skf.split(np.zeros(seq_len), y_np))
        except ValueError:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
            split_iter = list(kf.split(np.zeros(seq_len)))

        embeddings = []
        for train_index, test_index in split_iter:
            x_context = x[train_index]
            x_query = x[test_index]
            x_fold = torch.cat((x_context, x_query), dim=0).unsqueeze(0)
            y_context = torch.as_tensor(y_np[train_index], device=self.device).unsqueeze(0)
            with torch.no_grad():
                fold_emb = self.model.get_query_embedding(x_fold, y_context)
            embeddings.append(fold_emb.squeeze(0).cpu().numpy())

        return np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, self.model.d_model), dtype=np.float32)

    def _transform_test(self, x: torch.Tensor, y: torch.Tensor = None) -> np.ndarray:
        """
        Transform the test data.

        Parameters:
        x : tensor
            Input test data.
        y : tensor, optional
            Target values.

        Returns:
        np.ndarray
            Embeddings for the test data.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)

        train_x = self.x_train
        if train_x is None or self.y_train is None:
            raise ValueError("Embedder must be fitted before calling transform with mode='test'.")

        if train_x.shape[0] == 1 and x.shape[0] > 1:
            train_x = train_x.repeat(x.shape[0], 1, 1)
        elif train_x.shape[0] != x.shape[0]:
            raise ValueError("Mismatch between stored training batch and provided batch size.")

        batch_x = torch.cat([train_x, x.to(self.device)], dim=1)

        y_train = self.y_train
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(0)
        if y_train.shape[0] == 1 and x.shape[0] > 1:
            y_train = y_train.repeat(x.shape[0], 1)
        elif y_train.shape[0] != x.shape[0]:
            raise ValueError("Mismatch between stored labels and provided batch size.")

        with torch.no_grad():
            emb = self.model.get_query_embedding(batch_x, y_train)

        emb_np = emb.cpu().numpy()
        return emb_np.squeeze(0) if emb_np.shape[0] == 1 else emb_np

    def _transform_longitudinal(self, x: torch.Tensor, y: torch.Tensor = None, k: int = 5, fixed_window: bool = True) -> np.ndarray:
        """
        Transform the longitudinal data.

        Parameters:
        x : tensor
            Input longitudinal data.
        y : tensor, optional
            Target values.
        k : int, optional
            Number of previous points to use as context.
        fixed_window : bool, optional
            If True, keep only the previous k points; if False, keep all points before the current one.

        Returns:
        np.ndarray
            Embeddings for the longitudinal data.
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)

        seq_len = x.shape[0]
        if seq_len < 2:
            return np.zeros((0, self.model.d_model), dtype=np.float32)

        if fixed_window:
            window = min(k, seq_len - 1)
        else:
            window = None
        embeddings = []

        for i in range(1, seq_len):
            if fixed_window:
                assert window is not None
                start = max(0, i - window)
            else:
                start = 0
            context = x[start:i]
            if context.shape[0] == 0:
                continue

            query = x[i].unsqueeze(0)
            x_sample = torch.cat((context, query), dim=0).unsqueeze(0)
            y_dummy = torch.zeros((1, context.shape[0]), device=self.device)
            # print(x_sample.shape)

            with torch.no_grad():
                emb = self.model.get_query_embedding(x_sample, y_dummy)
            embeddings.append(emb.squeeze(0).squeeze(0).cpu().numpy())

        return np.stack(embeddings, axis=0) if embeddings else np.zeros((0, self.model.d_model), dtype=np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, mode: str = "test", **fit_params) -> np.ndarray:
        """
        Fit the model and then transform the input data.

        Parameters:
        X : array-like
            Input data to fit and transform.
        y : array-like, optional
            Target values.
        mode : str, optional
            Mode of transformation.

        Returns:
        np.ndarray
            Transformed data.
        """
        return self.fit(X, y).transform(X, mode=mode)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Parameters:
        deep : bool, optional
            If True, will return the parameters for this estimator and contained sub-objects.

        Returns:
        params : dict
            Parameter names mapped to their values.
        """
        return {"model_path": self.model_path, "device": self.device}

    def set_params(self, **params) -> 'APTEmbedder':
        """
        Set the parameters of this estimator.

        Parameters:
        **params : keyword arguments
            Parameters to set.

        Returns:
        self : object
            Returns the instance itself.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self