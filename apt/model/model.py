import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock
from .feedforward import FeedForward
from .utils import get_args, auc_metric

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)


class APT(nn.Module):
    def __init__(self, n_blocks, d_features, d_model=512, d_ff=2048, n_heads=4,
                 dropout=0.1, activation="gelu", norm_eps=1e-5, classification=True, n_classes=10):
        super().__init__()
        # TODO: feed normalized dataset in training, normalize dataset in inference
        # TODO: is the original zero initialization for transformer necessary?
        # TODO: dropout for context aggregation?
        self.init_args = get_args(vars())
        self.d_features = d_features
        self.classification = classification
        self.n_classes = n_classes

        self._emb_x = FeedForward(d_model, in_dim=d_features, out_dim=d_model,
            activation=activation, bias=True)
        self._emb_y = FeedForward(d_model, in_dim=1, out_dim=d_model,
            activation=activation, bias=True)

        self._transformer = nn.ModuleList(
            [TransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, activation=activation, norm_eps=norm_eps
            ) for _ in range(n_blocks)]
        )

        out_dim = n_classes if classification else 2
        self._out = FeedForward(d_model, out_dim=out_dim, activation=activation)

        self.x_train = None
        self.y_train = None
        self.feature_perm = None

    def forward(self, x, y_train, mask=None):
        split = y_train.shape[1]

        x = self._emb_x(x, mask=mask) # (batch_size, data_size, d_model)
        x_train, x_test = x[:, :split, ...], x[:, split:, ...] # (batch_size, n_train, d_model), (batch_size, n_test, d_model)
        y_train = self._emb_y(y_train.to(x.dtype).unsqueeze(-1)) # (batch_size, n_train, d_model)

        hidden = torch.cat([x_train + y_train, x_test], dim=1) # (batch_size, data_size, d_model)
        mask = self.get_mask(split, hidden.shape[1] - split).to(hidden.device)

        for _, block in enumerate(self._transformer):
            hidden = block(hidden, mask=mask) # (batch_size, data_size, d_model)

        h_test = hidden[:, split:, ...] # (batch_size, n_test, d_model)

        y_pred = self._out(h_test).squeeze(-1) # (batch_size, n_test)

        return y_pred

    def loss(self, x, y, split=None, mask=None, train_size=0.95):
        """
        x: (batch_size, data_size, sequence_length)
        y: (batch_size, data_size)
        mask: (batch_size, data_size, sequence_length)
        """
        if split is None:
            split = int(x.shape[1]*train_size)

        y_train, y_test = y[:, :split, ...], y[:, split:, ...] # (batch_size, n_train), (batch_size, n_test)

        out = self.forward(x, y_train, mask=mask)

        # prediction loss
        if self.classification:
            loss = self.classification_loss(out, y_test)
        else:
            loss = self.regression_loss(out, y_test)
        loss_dict = {"Prediction Loss": loss.item()}

        return loss, loss_dict

    def classification_loss(self, y_pred, y_test, eps=1e-4):
        probs = torch.softmax(y_pred, dim=-1)
        ce = -torch.log(torch.gather(probs, -1, y_test.unsqueeze(-1)) + eps)
        return ce.mean()

    def regression_loss(self, y_pred, y_test, eps=1e-6):
        return F.gaussian_nll_loss(
            y_pred[..., 0], y_test, F.softplus(y_pred[..., 1]), full=True, eps=eps
        )

    def get_mask(self, n_train, n_test):
        """
        enc mask:
            e.g. train - 4, test - 2
            [
             [1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0, 0],
            ]
        """
        mask = torch.cat((
            torch.ones(n_train+n_test, n_train),
            torch.zeros(n_train+n_test, n_test)
        ), dim=1) # (n_train+n_test, n_train+n_test)
        return mask

    @torch.no_grad()
    def predict_helper(self, x_train, y_train, x_test, batch_size=3000):
        """
        x_train: (train_size, sequence_length)
        y_train: (train_size,)
        x_test: (test_size, sequence_length)
        """
        device = next(self.parameters()).device

        train_size = min(batch_size, y_train.shape[0])
        x = torch.cat((x_train[:train_size, :], x_test), dim=-2)[:, :self.d_features]
        y_train = y_train[:train_size]

        x, y_train = map(lambda t: t.to(device), (x, y_train))
        out = self.forward(x.unsqueeze(0), y_train.unsqueeze(0)).squeeze(0)

        if self.classification:
            return torch.softmax(out[:, :(y_train.max() + 1)], dim=-1)
        return out[..., 0]

    @torch.no_grad()
    def evaluate_helper(self, x_train, y_train, x_test, y_test, batch_size=3000, metric=None):
        """
        x_train: (batch_size, train_size, sequence_length)
        y_train: (batch_size, train_size)
        x_test: (batch_size, test_size, sequence_length)
        y_test: (batch_size, test_size,)
        """
        device = next(self.parameters()).device
        y_test = y_test.to(device)
        target = y_test.cpu().numpy()

        y_pred = self.predict_helper(x_train, y_train, x_test, batch_size=batch_size)
        if self.classification:
            proba = y_pred.cpu().numpy()
            pred = torch.argmax(y_pred, dim=-1).cpu().numpy()
            if metric == "acc":
                return accuracy_score(target, pred)
            elif metric == "bacc":
                return balanced_accuracy_score(target, pred)
            elif metric == "ce":
                return log_loss(target, proba)
            elif metric == "auc":
                return auc_metric(target, proba)
            else:
                return {
                    "Test ACC": accuracy_score(target, pred),
                    "Test BACC": balanced_accuracy_score(target, pred),
                    "Test CE": log_loss(target, proba),
                    "Test AUC": auc_metric(target, proba),
                }
        pred = y_pred.cpu().numpy()
        if metric == "mse":
            return mean_squared_error(target, pred)
        elif metric == "mae":
            return mean_absolute_error(target, pred)
        elif metric == "r2":
            return r2_score(target, pred)
        else:
            return {
                "Test MSE": mean_squared_error(target, pred),
                "Test MAE": mean_absolute_error(target, pred),
                "Test R2": r2_score(target, pred),
            }

    def get_score(self, metric, value):
        if metric in ["ce", "mse", "mae"]:
            return -value
        else:
            return value

    @torch.no_grad()
    def fit(self, x_train, y_train, val_size=0.2, tune=True, metric=None, n_perms=32, batch_size=3000):
        """
        x_train: (train_size, sequence_length)
        y_train: (train_size)
        x_test: (test_size, sequence_length)
        y_test: (test_size,)
        """
        self.x_train = x_train
        self.y_train = y_train

        if tune:
            data_perm = torch.randperm(x_train.shape[0])
            val_size = int(val_size * x_train.shape[0])
            val_x_test = x_train[data_perm[:val_size]]
            val_x_train = x_train[data_perm[val_size:]]
            val_y_test = y_train[data_perm[:val_size]]
            val_y_train = y_train[data_perm[val_size:]]

            default_result = self.evaluate_helper(
                val_x_train, val_y_train, val_x_test, val_y_test,
                batch_size=batch_size, metric=metric
            )
            best_score = default_result["Test AUC"] if self.classification else -default_result["Test MSE"]
            best_perm = None
            for _ in range(n_perms):
                feature_perm = torch.randperm(x_train.shape[1])
                val_x_train_perm = val_x_train[:, feature_perm]
                val_x_test_perm = val_x_test[:, feature_perm]

                if metric is None:
                    metric = "auc" if self.classification else "mse"
                result = self.evaluate_helper(
                    val_x_train_perm, val_y_train, val_x_test_perm, val_y_test,
                    batch_size=batch_size, metric=metric
                )
                score = self.get_score(metric, result)

                if score > best_score:
                    best_perm = feature_perm
                    best_score = score
            self.feature_perm = best_perm

        return self

    def get_data(self, x_test):
        x_train = self.x_train
        y_train = self.y_train
        if self.feature_perm is not None:
            x_train = x_train[:, self.feature_perm]
            x_test = x_test[:, self.feature_perm]

        return x_train, y_train, x_test

    def evaluate(self, x_test, y_test, batch_size=3000, metric=None):
        return self.evaluate_helper(
            *self.get_data(x_test), y_test,
            batch_size=batch_size, metric=metric
        )

    def predict_proba(self, x_test, batch_size=3000):
        if self.classification:
            y_pred = self.predict_helper(*self.get_data(x_test), batch_size=batch_size)
            return y_pred.cpu().numpy()
        return NotImplementedError

    def predict(self, x_test, batch_size=3000):
        y_pred = self.predict_helper(*self.get_data(x_test), batch_size=batch_size)
        if self.classification:
            return torch.argmax(y_pred, dim=-1).cpu().numpy()
        return y_pred.cpu().numpy()
