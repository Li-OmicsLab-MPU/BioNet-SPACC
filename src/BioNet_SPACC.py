import os
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
class BioNetSpaccClustering:
    def __init__(self, config_path=ROOT_DIR/'config'/'config.yaml'):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = ROOT_DIR / self.config['data']['output_dir']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu_kmeans = False
        try:
            import cupy as cp
            from cuml.cluster import KMeans as KMeans_GPU
            from torch.utils.dlpack import to_dlpack
            from cupy import from_dlpack
            self.cp = cp
            self.KMeans_GPU = KMeans_GPU
            self.to_dlpack = to_dlpack
            self.from_dlpack = from_dlpack
            self.use_gpu_kmeans = True
            print("[Acceleration] RAPIDS (cuML/CuPy) detected. Full-pipeline GPU acceleration enabled.")
        except ImportError:
            print(f"[Warning] RAPIDS environment not detected. Falling back to CPU mode.")

    def load_data(self):
        file_path = self.output_dir/"1.X_train_final.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError

        self.X_discovery = pd.read_csv(file_path, index_col=0)
        self.A_matrix = np.load(f"{self.output_dir}/2.Adjacency_Matrix.npy")
        self.L_matrix = np.load(f"{self.output_dir}/2.Laplacian_Matrix.npy")
        n_features_data = self.X_discovery.shape[1]
        n_features_net = self.A_matrix.shape[0]

        if n_features_data != n_features_net:
            raise ValueError(f"Dimensionality mismatch: Number of features in the expression matrix ({n_features_data}) "
                             f"does not match the number of nodes in the PPI network ({n_features_net})")
        self.X_discovery = self.X_discovery.sort_index(axis=1)

    def netspacc_optimization(self, X, L, k, s, lambda2, max_iter=100, rho=1.0):
        """
        Standard Net-Spacc: Alternating optimization of feature weights (w) and cluster centroids.
        Objective: max w^T * BCSS - λ2 * w^T * L * w
        Net-Spacc ADMM Optimization (Strict replication of Witten's framework + Network Penalty)

        Objective Function:
             max_w  w^T * bcss - lambda2 * w^T * L * w
             s.t.   ||w||_2 = 1,  ||w||_1 <= s,  w >= 0

        Parameters:
             X: ndarray (n_samples, n_features)
             L: ndarray (n_features, n_features), Laplacian matrix
             k: int, number of clusters
             s: float, L1 sparsity constraint (only s is used, lambda1 removed)
             lambda2: float, network smoothing penalty
             rho: float, ADMM penalty parameter

        Returns:
             w: ndarray, feature weights
             labels: ndarray, cluster labels
        """

        n, p = X.shape
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        L_torch = torch.tensor(L, dtype=torch.float32, device=self.device)
        A = 2 * lambda2 * L_torch + (rho + 1e-4) * torch.eye(p, device=self.device)
        try:
            L_chol = torch.linalg.cholesky(A + 1e-6 * torch.eye(p, device=self.device))
        except RuntimeError as e:
            A += 1e-3 * torch.eye(p, device=self.device)
            L_chol = torch.linalg.cholesky(A)

        w = torch.ones(p, device=self.device) / np.sqrt(p)
        z = w.clone()
        u = torch.zeros(p, device=self.device)

        for iteration in range(max_iter):
            X_weighted = X_torch * torch.sqrt(w.clamp(min=0))

            if self.use_gpu_kmeans:
                X_cupy = self.from_dlpack(self.to_dlpack(X_weighted))
                labels = self.KMeans_GPU(n_clusters=k, max_iter=10).fit_predict(X_cupy)
                if hasattr(labels, 'get'):
                    labels = labels.get()
            else:
                labels = KMeans(
                    n_clusters=k,
                    n_init=10,
                    max_iter=300,
                    random_state=42
                ).fit_predict(X_weighted.cpu().numpy())

            bcss = self._compute_bcss(X_torch, labels, k)
            w_new, z_new, u_new = self._admm_update_w_optimized(bcss, L_chol, w, z, u, s, rho)
            primal_residual = torch.norm(w_new - z_new).item()
            dual_residual = torch.norm(rho * (z_new - z)).item()
            if iteration > 5 and primal_residual < 1e-4 and dual_residual < 1e-4:
                break
            w, z, u = w_new, z_new, u_new

        X_final_weighted = X_torch * torch.sqrt(z.clamp(min=0))
        if self.use_gpu_kmeans:
            X_cupy = self.from_dlpack(self.to_dlpack(X_final_weighted))
            labels_final = self.KMeans_GPU(n_clusters=k, max_iter=300).fit_predict(X_cupy)
            if hasattr(labels_final, 'get'):
                labels_final = labels_final.get()
        else:
            labels_final = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42).fit_predict(
                X_final_weighted.cpu().numpy())

        return z.cpu().numpy(), labels_final

    def _compute_bcss(self, X, labels, k):
        if not isinstance(labels, (np.ndarray, list)):
            if hasattr(labels, 'get'):
                labels = labels.get()
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        else:
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        n_samples = X.shape[0]
        global_mean = X.mean(dim=0)
        bcss = torch.zeros(X.shape[1], device=self.device)
        for cluster_id in range(k):
            mask = (labels == cluster_id)
            if mask.sum() == 0:
                continue
            cluster_mean = X[mask].mean(dim=0)
            n_k = mask.sum()
            bcss += n_k * (cluster_mean - global_mean) ** 2

        return bcss / n_samples

    def _admm_update_w_optimized(self, bcss, L_chol, w, z, u, s, rho):
        """
        Three-step ADMM update (lambda1 removed, strictly enforcing L1 constraint s)
        Objective: max w^T * bcss - lambda2 * w^T * L * w
        Subject to: ||w||_2 = 1,  ||z||_1 <= s,  w = z
        ADMM for max w^T·bcss - λ₂·w^T·L·w
        Conversion: max f(w) ≡ min -f(w)
        Augmented Lagrangian:
            L(w,z,u) = -w^T·bcss + λ₂·w^T·L·w + (ρ/2)||w-z+u||²_2
        Closed-form solution (Boyd et al. 2011, Sec 4.2):
            (2λ₂L + ρI)w = bcss + ρ(z-u)
        """
        b = bcss + rho * (z - u)
        b = b.view(-1, 1)
        w_new = torch.cholesky_solve(b, L_chol).flatten()
        w_new = w_new / (torch.norm(w_new) + 1e-8)
        v = w_new + u
        z_new = self._project_l1_ball(v, s)
        u_new = u + (w_new - z_new)

        return w_new, z_new, u_new

    def _project_l1_ball(self, z, s):
        z = torch.clamp(z, min=0)
        if torch.sum(z) <= s:
            return z

        z_sorted, _ = torch.sort(z, descending=True)
        cumsum = torch.cumsum(z_sorted, dim=0)
        k = torch.arange(1, len(z) + 1, device=self.device)
        condition = z_sorted > (cumsum - s) / k
        if condition.any():
            k_max = torch.where(condition)[0][-1].item() + 1
            theta = (cumsum[k_max - 1] - s) / k_max
        else:
            return torch.zeros_like(z)
        z_proj = torch.clamp(z - theta, min=0)
        return z_proj

    def calculate_pac(self, M, lower=0.1, upper=0.9):
        consensus_values = M[np.triu_indices_from(M, k=1)]
        ambiguous = np.sum((consensus_values > lower) & (consensus_values < upper))
        pac = ambiguous / len(consensus_values)
        return pac