import torch
import pandas as pd
import numpy as np

class ALS:
    def __init__(self, n_users:int, n_items:int, rank:int=10, lamb:float=5, max_iter:int=100, device=None):
        """Initialize ALS model with users, items, rank, regularization, iterations, and device."""
        
        if not isinstance(n_users, int) or n_users <= 0:
            raise ValueError(f"n_users must be a positive integer, got {n_users}")
        self.n_users = n_users

        if not isinstance(n_items, int) or n_items <= 0:
            raise ValueError(f"n_items must be a positive integer, got {n_items}")
        self.n_items = n_items

        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")
        self.rank = rank

        if not isinstance(lamb, (int, float)) or lamb < 0:
            raise ValueError(f"lambda must be a non-negative number, got {lamb}")
        self.lamb = float(lamb)

        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter}")
        
        self.max_iter = max_iter
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.X = torch.randn(rank, n_users, device=self.device, dtype=torch.float32)
        self.Y = torch.randn(rank, n_items, device=self.device, dtype=torch.float32)

        self.lam_eye = torch.eye(rank, device=self.device) * lamb

    @torch.no_grad()
    def _update_users(self, X: torch.Tensor, Y: torch.Tensor, u_idx: torch.Tensor, i_idx: torch.Tensor, r_vals: torch.Tensor, lam_eye: torch.Tensor) -> torch.Tensor:
        """Update user latent factors using ALS optimization step."""
       
        S = torch.einsum('am,bm->mab', Y, Y)
        rank = X.size(0)
        n = X.size(1)

        A = torch.zeros((n, rank, rank), device=X.device)
        A.index_add_(dim=0, index=u_idx, source=S.index_select(0, i_idx))
        A += lam_eye.unsqueeze(0)

        b = torch.zeros((rank, n), dtype=X.dtype, device=X.device)
        b.index_add_(dim=1, index=u_idx, source=Y.index_select(1, i_idx) * r_vals.unsqueeze(0))

        L = torch.linalg.cholesky(A)
        X_T = torch.cholesky_solve(b.T.unsqueeze(-1), L).squeeze(-1)
        return X_T.T

    @torch.no_grad()
    def _update_items(self, X: torch.Tensor, Y: torch.Tensor, u_idx: torch.Tensor, i_idx: torch.Tensor, r_vals: torch.Tensor, lam_eye: torch.Tensor) -> torch.Tensor:
        """Update item latent factors using ALS optimization step."""
        
        S = torch.einsum('am,bm->mab', X, X)
        rank = Y.size(0)
        m = Y.size(1)

        A = torch.zeros((m, rank, rank), device=Y.device)
        A.index_add_(dim=0, index=i_idx, source=S.index_select(0, u_idx))
        A += lam_eye.unsqueeze(0)

        b = torch.zeros((rank, m), dtype=Y.dtype, device=Y.device)
        b.index_add_(dim=1, index=i_idx, source=X.index_select(1, u_idx) * r_vals.unsqueeze(0))

        L = torch.linalg.cholesky(A)
        Y_T = torch.cholesky_solve(b.T.unsqueeze(-1), L).squeeze(-1)
        return Y_T.T

    @torch.no_grad()
    def _rmse(self, X: torch.Tensor, Y: torch.Tensor, u_idx: torch.Tensor, i_idx: torch.Tensor, r_vals: torch.Tensor) -> float:
        """Compute RMSE between predicted and actual ratings."""
        
        pred = (X[:, u_idx] * Y[:, i_idx]).sum(dim=0)
        rmse = torch.sqrt(torch.mean((pred - r_vals) ** 2))
        return float(rmse.detach().cpu())

    def fit(self,*, df_users: pd.Series, df_items: pd.Series, df_ratings: pd.Series):
        """Train ALS model by alternating updates of user and item factors until convergence."""

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        u_idx = torch.tensor(df_users.values, dtype=torch.long, device=self.device)
        i_idx = torch.tensor(df_items.values, dtype=torch.long, device=self.device)
        r_vals = torch.tensor(df_ratings.values, dtype=torch.float32, device=self.device)

        prev_loss = float('inf')
        iter_count = 0

        while True:
            self.X = self._update_users(self.X, self.Y, u_idx, i_idx, r_vals, self.lam_eye)
            self.Y = self._update_items(self.X, self.Y, u_idx, i_idx, r_vals, self.lam_eye)

            rmse = self._rmse(self.X, self.Y, u_idx, i_idx, r_vals)
            print(f'Iteration {iter_count}: RMSE = {rmse:.4f}')

            iter_count += 1
            if abs(rmse - prev_loss) < 1e-3 or iter_count >= self.max_iter:
                break
            prev_loss = rmse

        return self.X, self.Y
