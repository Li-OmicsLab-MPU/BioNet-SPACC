"""
Stage 0: PPI Network Construction (Based on Public Prior Knowledge)
- Load the preprocessed protein list
- Match against STRING or other PPI databases
- Output the adjacency matrix and Laplacian matrix
"""

import os
import yaml
import numpy as np
import pandas as pd

from utils.network import (
    load_ppi_network,
    build_adjacency_matrix,
    compute_graph_laplacian,
)
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent

class Stage2PPINetwork:
    def __init__(self, config_path=ROOT_DIR/'config'/'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = ROOT_DIR / self.config['data']['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_network(self):
        train_file = self.output_dir/"1.X_train_final.csv"
        if not os.path.exists(train_file):
            raise FileNotFoundError
        df_train = pd.read_csv(train_file, index_col=0)
        kept_proteins = sorted(list(df_train.columns))
        ppi_file = ROOT_DIR / self.config['data']['ppi_file']
        threshold = self.config['qc']['ppi_score_threshold']
        ppi_edges = load_ppi_network(ppi_file, score_threshold=threshold)
        expr_nodes = set(kept_proteins)
        ppi_nodes = set()

        if isinstance(ppi_edges, pd.DataFrame):
            if 'protein1' in ppi_edges.columns:
                ppi_nodes.update(ppi_edges['protein1'].dropna().astype(str).tolist())
                ppi_nodes.update(ppi_edges['protein2'].dropna().astype(str).tolist())
            elif 'Source' in ppi_edges.columns:
                ppi_nodes.update(ppi_edges['Source'].dropna().astype(str).tolist())
                ppi_nodes.update(ppi_edges['Target'].dropna().astype(str).tolist())
        elif hasattr(ppi_edges, 'nodes'):
            ppi_nodes.update(ppi_edges.nodes())
        elif isinstance(ppi_edges, list):
            for edge in ppi_edges:
                if isinstance(edge, dict):
                    ppi_nodes.add(edge.get('Source') or edge.get('protein1') or edge.get('node1'))
                    ppi_nodes.add(edge.get('Target') or edge.get('protein2') or edge.get('node2'))
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    ppi_nodes.add(edge[0])
                    ppi_nodes.add(edge[1])

        ppi_nodes = {str(node).strip().upper() for node in ppi_nodes if node and str(node).strip() != 'NONE'}
        overlap_nodes = expr_nodes.intersection(ppi_nodes)
        missing_nodes = expr_nodes - ppi_nodes

        if len(missing_nodes) > 0:
            missing_df = pd.DataFrame({'Unmapped_Proteins_in_Expression': list(missing_nodes)})
            missing_df.to_csv(f"{self.output_dir}/2.0.PPI_unmapped_names_check.csv", index=False)
        self.A_matrix = build_adjacency_matrix(ppi_edges, kept_proteins)
        self.L_matrix = compute_graph_laplacian(self.A_matrix, normalized=True)
        isolated_count = np.sum(self.A_matrix.sum(axis=1) == 0)
        isolated_ratio = isolated_count / len(kept_proteins)
        np.save(f"{self.output_dir}/2.Adjacency_Matrix.npy", self.A_matrix)
        np.save(f"{self.output_dir}/2.Laplacian_Matrix.npy", self.L_matrix)
        isolated_proteins = [p for i, p in enumerate(kept_proteins) if self.A_matrix[i].sum() == 0]

        with open(f"{self.output_dir}/2.isolated_proteins.txt", 'w') as f:
            f.write('\n'.join(isolated_proteins))

if __name__ == "__main__":
    ppi_builder = Stage2PPINetwork()
    ppi_builder.build_network()