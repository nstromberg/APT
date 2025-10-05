from sklearn.base import TransformerMixin, BaseEstimator
import torch
import numpy as np
from apt.model import APT
from sklearn.model_selection import StratifiedKFold  # Add this import
import pathlib
import os

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
            import requests
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
        self.x_train = X
        if y is None:
            self.y_train = np.zeros(len(X))
        else:
            self.y_train = y
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
        if y is None:
            y = np.zeros(x.shape[0])
        y = torch.as_tensor(y).to(self.device)
        x = torch.as_tensor(x).to(self.device)

        seq_len = x.shape[0]
        skf = StratifiedKFold(n_splits=k_folds)
        x_folds = []
        y_folds = []
        for train_index, test_index in skf.split(np.zeros(seq_len), y.cpu().numpy()):
            x_folds.append(torch.concat((x[train_index], x[test_index]), dim=0))
            y_folds.append(y[train_index])
        
        # Combine all training data and test data for the current fold
        x_batch = torch.stack(x_folds).float()  # (k_folds, n_steps, n_features)
        y_batch = torch.stack(y_folds).float()       # (k_folds, n_steps - fold_size)

        with torch.no_grad():
            emb = torch.flatten(self.model.get_query_embedding(x_batch, y_batch), end_dim=-2)
        return emb.cpu().numpy()

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
        batch_x = torch.cat([self.x_train, x], dim=1)
        with torch.no_grad():
            emb = self.model.get_query_embedding(batch_x, self.y_train).squeeze()
        return emb.cpu().numpy()

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
        embeddings = []
        embeddings.append(self._transform_train(x[:k], k_folds=k))  # Initial embedding for first k points
        x_batch = []
        y_batch = []  # Initialize y_batch to collect dummy labels
        for i in range(k, x.shape[0]):
            if fixed_window:
                # Use the previous k points as context
                x_long = torch.cat([x[i-k:i], x[i].unsqueeze(0)], dim=0)  # Change to (k+1, n_features)
            else:
                # Use all points before the current one
                x_long = x[:i+1]  # Change to (i+1, n_features)
            x_batch.append(x_long)

            # Create dummy labels for y
            y_dummy = torch.zeros(k).to(self.device)
            y_batch.append(y_dummy)  # Collect dummy labels
        
        # Call get_query_embedding once with the constructed batch
        x_batch_tensor = torch.stack(x_batch).float()  # (num_batches, k+1, n_features) or (num_batches, i+1, n_features)
        y_batch_tensor = torch.stack(y_batch).float()  # (num_batches, k)
        with torch.no_grad():
            emb = self.model.get_query_embedding(x_batch_tensor, y_batch_tensor).squeeze()
        embeddings.append(emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, mode: str = "test") -> np.ndarray:
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