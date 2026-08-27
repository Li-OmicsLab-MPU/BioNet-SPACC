import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from BioNet_SPACC import BioNetSpaccClustering
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent

class NetSpaccFinalClustering(BioNetSpaccClustering):
    def __init__(self):
        super().__init__()
        self.final_cluster_dir = ROOT_DIR / self.config['data']['final_cluster_dir']
        os.makedirs(self.final_cluster_dir, exist_ok=True)

    def bootstrap_consensus_matrix(self, X, L, k, s, lambda2, n_bootstrap=100, subsample_ratio=0.8, ref_labels=None,
                                   ref_w=None):
        rng = np.random.RandomState(42)
        n_samples = X.shape[0]
        subsample_size = int(subsample_ratio * n_samples)

        M = np.zeros((n_samples, n_samples))
        I = np.zeros((n_samples, n_samples))

        w_all = []
        ari_scores = []
        nmi_scores = []
        sil_scores = []
        jaccard_scores = []
        n_feats_list = []

        if ref_w is not None:
            baseline_features = set(np.where(ref_w > 0.01)[0])

        for b in tqdm(range(n_bootstrap), desc=f"Subsampling K={k}"):
            idx = rng.choice(n_samples, size=subsample_size, replace=False)
            X_boot = X[idx, :]

            z_boot, labels_boot = self.netspacc_optimization(X_boot, L, k, s, lambda2)
            w_all.append(z_boot)

            if hasattr(labels_boot, 'get'):
                labels_boot = labels_boot.get()

            if ref_labels is not None:
                ref_labels_subset = ref_labels[idx]
                ari = adjusted_rand_score(ref_labels_subset, labels_boot)
                ari_scores.append(ari)

                nmi = normalized_mutual_info_score(ref_labels_subset, labels_boot)
                nmi_scores.append(nmi)
                try:
                    sil = silhouette_score(X_boot, labels_boot)
                    sil_scores.append(sil)
                except Exception:
                    sil_scores.append(np.nan)

            if ref_w is not None:
                boot_w_cpu = z_boot.get() if hasattr(z_boot, 'get') else z_boot
                boot_features = set(np.where(boot_w_cpu > 0.01)[0])
                n_feats_list.append(len(boot_features))
                if len(baseline_features | boot_features) > 0:
                    jaccard = len(baseline_features & boot_features) / len(baseline_features | boot_features)
                else:
                    jaccard = 0.0
                jaccard_scores.append(jaccard)

            sub_conn = (labels_boot[:, None] == labels_boot[None, :]).astype(float)
            grid = np.ix_(idx, idx)
            M[grid] += sub_conn
            I[grid] += 1

        consensus = np.divide(M, I, out=np.zeros_like(M), where=I > 0)
        np.fill_diagonal(consensus, 1.0)
        selection_freq = np.mean(np.array(w_all) > 0.01, axis=0)

        return consensus, selection_freq, ari_scores, nmi_scores, sil_scores, jaccard_scores, n_feats_list

    def final_clustering(self, params):
        """Perform final clustering using optimal parameters"""

        X = self.X_discovery.values
        L = self.L_matrix
        ref_w, ref_labels = self.netspacc_optimization(X, L, int(params['K']), params['s'], params['lambda2'])
        if hasattr(ref_labels, 'get'):
            ref_labels = ref_labels.get()
            ref_w = ref_w.get() if hasattr(ref_w, 'get') else ref_w

        from sklearn.cluster import SpectralClustering
        M, selection_freq, ari_scores, nmi_scores, sil_scores, jaccard_scores, n_feats_list = self.bootstrap_consensus_matrix(
            X, L, k=int(params['K']), s=params['s'],
            lambda2=params['lambda2'], n_bootstrap=1000, ref_labels=ref_labels, ref_w=ref_w
        )
        stability_raw_df = pd.DataFrame({
            'ARI_Scores': ari_scores,
            'NMI_Scores': nmi_scores,
            'Sil_scores': sil_scores,
            'Jaccard_Scores': jaccard_scores,
        })
        stability_raw_df.to_csv(ROOT_DIR / self.final_cluster_dir/"Discovery_Stability_Iteration_Scores.csv", index=False)
        pac = self.calculate_pac(M)
        labels_final = SpectralClustering(
            n_clusters=int(params['K']),
            affinity='precomputed',
            random_state=42
        ).fit_predict(M)
        w_final = selection_freq

        self.labels_discovery = pd.Series(labels_final, index=self.X_discovery.index, name='Subtype')
        self.labels_discovery.to_csv(ROOT_DIR / self.final_cluster_dir/"Y_Labels_Discovery.csv")

        core_proteins_idx = np.where(w_final >= 0.2)[0]
        self.core_proteins_df = pd.DataFrame({
            'Protein': self.X_discovery.columns[core_proteins_idx],
            'Weight': w_final[core_proteins_idx]
        }).sort_values('Weight', ascending=False)
        self.core_proteins_df.to_csv(ROOT_DIR / self.final_cluster_dir/"Core_Proteins.csv", index=False)

        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
        X_weighted = X * np.sqrt(np.maximum(w_final, 0))
        silhouette = silhouette_score(X_weighted, labels_final)
        calinski = calinski_harabasz_score(X_weighted, labels_final)
        davies = davies_bouldin_score(X_weighted, labels_final)

        pd.DataFrame([{
            'Silhouette': silhouette,
            'Calinski_Harabasz': calinski,
            'Davies_Bouldin': davies,
            'PAC': pac,
            'K': int(params['K']),
            'N_Core_Proteins': len(self.core_proteins_df)
        }]).to_csv(ROOT_DIR / self.final_cluster_dir/"clustering_diagnostics.csv", index=False)

    def run(self):
        self.load_data()

        final_params = {
            'K': 2,
            's': 5.0,
            'lambda2': 1.0,
        }
        print(f"Parameters: K={final_params['K']}, s={final_params['s']}, λ2={final_params['lambda2']}")
        self.final_clustering(final_params)

if __name__ == "__main__":
    clusterer = NetSpaccFinalClustering()
    clusterer.run()