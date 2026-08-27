import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import yaml
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent

class StructuralValidation:
    def __init__(self,
                 config_path=ROOT_DIR/'config'/'config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.output_dir = ROOT_DIR / self.config['data']['output_dir']
        self.final_cluster_dir = ROOT_DIR / self.config['data']['final_cluster_dir']
        self.results_dir4a = ROOT_DIR / self.config['data']['results_dir4a']
        os.makedirs(self.results_dir4a, exist_ok=True)

    def load_data(self):

        X_val_full = pd.read_csv(f"{self.output_dir}/1.X_test_final.csv", index_col=0)
        core_proteins = pd.read_csv(f"{self.final_cluster_dir}/Core_Proteins.csv")
        self.X_validation = X_val_full[core_proteins['Protein']]
        self.labels_discovery = pd.read_csv(
            f"{self.final_cluster_dir}/Y_Labels_Discovery.csv", index_col=0
        )
        self.K = self.labels_discovery['Subtype'].nunique()

        print(f"Validation: {self.X_validation.shape}")
        print(f"core proteins: {len(core_proteins)}")

    def _perform_clustering(self, data):
        S = cosine_similarity(data)
        S = (S + 1) / 2
        labels = SpectralClustering(
            n_clusters=self.K,
            affinity='nearest_neighbors',
            random_state=42,
            n_init=10
        ).fit_predict(S)
        return labels

    def reclustering(self):
        labels_val = self._perform_clustering(self.X_validation)
        self.labels_validation = pd.Series(
            labels_val,
            index=self.X_validation.index,
            name='Subtype'
        )
        self.real_sil = silhouette_score(self.X_validation, labels_val)
        print(f"Cluster distribution: {self.labels_validation.value_counts().to_dict()}")
        print(f"Validation Real Silhouette: {self.real_sil:.4f}")

    def run_permutation_test(self, n_permutations=1000):
        np.random.seed(42)
        print(f"\n(Permutations={n_permutations})...")

        null_scores = []
        X_mat = self.X_validation.values

        for i in tqdm(range(n_permutations), desc="Permuting"):
            X_perm = np.apply_along_axis(np.random.permutation, 0, X_mat)
            try:
                S_perm = cosine_similarity(X_perm)
                S_perm = (S_perm + 1) / 2
                labels_perm = SpectralClustering(
                    n_clusters=self.K,
                    affinity='nearest_neighbors',
                    random_state=None,
                    n_init=5
                ).fit_predict(S_perm)
                sil_perm = silhouette_score(X_perm, labels_perm)
                null_scores.append(sil_perm)
            except:
                continue

        null_scores = np.array(null_scores)
        p_val = (np.sum(null_scores >= self.real_sil) + 1) / (len(null_scores) + 1)
        print(f"  -> Null Silhouette Mean: {np.mean(null_scores):.4f} ± {np.std(null_scores):.4f}")
        print(f"  -> Real Silhouette: {self.real_sil:.4f}")
        print(f"  -> P-value: {p_val:.2e} {'***' if p_val < 0.001 else ''}")
        pd.DataFrame({'Null_Scores': null_scores}).to_csv(f"{self.results_dir4a}/Permutation_Scores.csv")

    def align_labels(self):
        print("\nCentroid based label mapping")
        X_disc = pd.read_csv(f"{self.output_dir}/1.X_train_final.csv", index_col=0)
        core_proteins = pd.read_csv(f"{self.final_cluster_dir}/Core_Proteins.csv")
        X_disc_core = X_disc[core_proteins['Protein']]
        centroids_disc = {}
        y_disc = pd.read_csv(f"{self.final_cluster_dir}/Y_Labels_Discovery.csv", index_col=0)
        for k in range(self.K):
            mask = y_disc['Subtype'] == k
            centroids_disc[k] = X_disc_core.loc[mask].mean(axis=0).values
        centroids_val = {}
        for k in range(self.K):
            mask = self.labels_validation == k
            centroids_val[k] = self.X_validation.loc[mask].mean(axis=0).values
        similarity_matrix = np.zeros((self.K, self.K))

        for i in range(self.K):
            for j in range(self.K):
                similarity_matrix[i, j] = cosine_similarity(
                    centroids_disc[i].reshape(1, -1),
                    centroids_val[j].reshape(1, -1)
                )[0, 0]
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        label_mapping = {col_ind[i]: i for i in range(self.K)}

        print("Label mapping relationship:")
        for val_label, disc_label in label_mapping.items():
            sim = similarity_matrix[disc_label, val_label]
            print(f"  Validation Subtype{val_label} → Discovery Subtype{disc_label} (similarity={sim:.3f})")

        self.labels_validation = self.labels_validation.map(label_mapping)
        print(f"Distribution after alignment: {self.labels_validation.value_counts().to_dict()}")

    def save_results(self):
        self.labels_validation.to_csv(
            f"{self.final_cluster_dir}/Y_Labels_Validation.csv")

    def run(self):
        self.load_data()
        self.reclustering()
        self.run_permutation_test(n_permutations=1000)
        self.align_labels()
        self.save_results()

if __name__ == "__main__":
    stage4a = StructuralValidation()
    stage4a.run()