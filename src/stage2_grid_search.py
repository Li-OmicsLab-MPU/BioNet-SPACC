"""
Net-Spacc Clustering (ADMM Optimization + Closed-loop Biological Validation)
- Hyperparameter grid search (ADMM optimization)
- Automated candidate parameter screening
"""
import os
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from BioNet_SPACC import BioNetSpaccClustering

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
class NetSpaccGridSearch(BioNetSpaccClustering):
    def __init__(self):
        super().__init__()
        self.cluster_dir = ROOT_DIR / self.config['data']['cluster_dir']
        os.makedirs(self.cluster_dir, exist_ok=True)

    def bootstrap_consensus_matrix(self, X, L, k, s, lambda2, n_bootstrap=100, subsample_ratio=0.8):
        rng = np.random.RandomState(42)
        n_samples = X.shape[0]
        subsample_size = int(subsample_ratio * n_samples)
        if subsample_size < 50:
            raise ValueError(f"Sample size after subsampling({subsample_size}) < 50. Consider increasing the sample size or using sampling with replacement.")

        M = np.zeros((n_samples, n_samples))
        I = np.zeros((n_samples, n_samples))

        w_all = []

        for b in tqdm(range(n_bootstrap), desc=f"Subsampling K={k}"):
            idx = rng.choice(n_samples, size=subsample_size, replace=False)
            X_boot = X[idx, :]
            z_boot, labels_boot = self.netspacc_optimization(X_boot, L, k, s, lambda2)
            w_all.append(z_boot)

            if hasattr(labels_boot, 'get'):
                labels_boot = labels_boot.get()

            sub_conn = (labels_boot[:, None] == labels_boot[None, :]).astype(float)
            grid = np.ix_(idx, idx)
            M[grid] += sub_conn
            I[grid] += 1

        consensus = np.divide(M, I, out=np.zeros_like(M), where=I > 0)
        np.fill_diagonal(consensus, 1.0)
        selection_freq = np.mean(np.array(w_all) > 0.01, axis=0)
        return consensus, selection_freq

    def grid_search(self):
        print("\n[Hyperparameter Grid Search]")

        X = self.X_discovery.values
        L = self.L_matrix

        k_range = self.config['netspacc']['k_range']
        s_range = self.config['netspacc']['s_range']
        lambda2_range = self.config['netspacc']['lambda2_range']
        n_bootstrap = self.config['netspacc']['bootstrap_n']
        results = []

        for k in k_range:
            for s, lambda2 in product(s_range, lambda2_range):
                print(f"\nTesting: K={k}, s={s}, λ2={lambda2}")

                M, selection_freq = self.bootstrap_consensus_matrix(X, L, k, s, lambda2, n_bootstrap=n_bootstrap)
                pac = self.calculate_pac(M)
                results.append({
                    'K': k,
                    's': s,
                    'lambda2': lambda2,
                    'PAC': pac,
                    'n_selected_features': np.sum(selection_freq >= 0.2)
                })
                print(f"  PAC={pac:.4f}, Number of selected features={np.sum(selection_freq >= 0.2)}")
                if pac > 0.5:
                    print(f" Warning: PAC={pac:.2f} exceeds the threshold, partition may be unstable.")

        self.results_df = pd.DataFrame(results)
        self.results_df.to_csv(
            f"{self.cluster_dir}/netspacc_admm_grid_search.csv",
            index=False
        )

    def select_automated_candidates(self):
        """
        [Customized Version] Screen 5 sets of candidate parameters for full comparison (Table).
        1. Clinical_Base: 40-100 features, λ2=0 (Control)
        2. Clinical_Net:  40-100 features, λ2≥0.05 (Primary)
        3. Broad_Base:    >100 features,   λ2=0 (Control)
        4. Broad_Net:     >100 features,   λ2≥0.05 (Mechanistic)
        5. Global_Stable: Lowest global PAC (Fallback/Safeguard)
        """
        df = self.results_df[
            (self.results_df['n_selected_features'] > 0) &
            (self.results_df['PAC'] < 0.25)
            ].copy()

        if df.empty:
            raise ValueError("No parameters meet the basic requirements (Features>0 & PAC<0.25)")

        candidates = []
        def add_candidate(condition, name, desc):
            subset = df[condition]
            if not subset.empty:
                best = subset.loc[subset['PAC'].idxmin()]
                if any(c['K'] == int(best['K']) and c['s'] == float(best['s']) and c['lambda2'] == float(
                        best['lambda2']) for c in candidates):
                    return
                candidates.append({
                    'name': name,
                    'K': int(best['K']), 's': float(best['s']), 'lambda2': float(best['lambda2']),
                    'PAC': best['PAC'], 'Features': int(best['n_selected_features'])
                })
                print(
                    f"Selected[{name:<13}] ({desc}): K={int(best['K'])}, s={best['s']}, λ2={best['lambda2']} -> PAC={best['PAC']:.4f}")

        add_candidate((df['n_selected_features'] >= 40) & (df['n_selected_features'] <= 100) & (df['lambda2'] == 0),
                      'Clinical_Base', 'Feat 40-100, λ2=0')
        add_candidate((df['n_selected_features'] >= 40) & (df['n_selected_features'] <= 100) & (df['lambda2'] >= 0.05),
                      'Clinical_Net', 'Feat 40-100, λ2≥0.05')
        add_candidate((df['n_selected_features'] > 100) & (df['lambda2'] == 0),
                      'Broad_Base', 'Feat >100, λ2=0')
        add_candidate((df['n_selected_features'] > 100) & (df['lambda2'] >= 0.05),
                      'Broad_Net', 'Feat >100, λ2≥0.05')
        best_stable = df.loc[df['PAC'].idxmin()]

        if not any(c['K'] == int(best_stable['K']) and c['s'] == float(best_stable['s']) and c['lambda2'] == float(
                best_stable['lambda2']) for c in candidates):
            candidates.append({
                'name': 'Global_Stable',
                'K': int(best_stable['K']), 's': float(best_stable['s']), 'lambda2': float(best_stable['lambda2']),
                'PAC': best_stable['PAC'], 'Features': int(best_stable['n_selected_features'])
            })
        else:
            print(f"[Global_Stable] is already included in the groups above, skipping addition.")

        return candidates

    def determine_final_best(self, candidates):
        report_data = []
        for res in candidates:
            report_data.append({
                'Mode': res['name'],
                'K': res['K'], 's': res['s'], 'lambda2': res['lambda2'],
                'Features': res['Features'],
                'PAC': res['PAC'],
            })
        df_report = pd.DataFrame(report_data)
        print(df_report.to_string(index=False))
        df_report.to_csv(f"{self.cluster_dir}/Candidate_Comparison_Report.csv", index=False)
        print(f"\nDetailed report saved to:{self.cluster_dir}/Candidate_Comparison_Report.csv")
        return None

    def run(self):
        """Execute the full pipeline"""

        self.load_data()
        csv_path = f"{self.cluster_dir}/netspacc_admm_grid_search.csv"
        if os.path.exists(csv_path):
            print(f"[Notice] Detected existing grid search results:{csv_path}. Loading directly...")
            print(f"{self.cluster_dir}/netspacc_admm_grid_search.csv")
            self.results_df = pd.read_csv(csv_path)
        else:
            self.grid_search()
        candidates = self.select_automated_candidates()
        self.determine_final_best(candidates)

if __name__ == "__main__":
    searcher = NetSpaccGridSearch()
    searcher.run()