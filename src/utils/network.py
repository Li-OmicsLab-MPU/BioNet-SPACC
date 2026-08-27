"""
PPI network processing tool
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian


def load_ppi_network(ppi_file, score_threshold=700):
    ppi = pd.read_csv(ppi_file, sep='\t')
    ppi_filtered = ppi[ppi['combined_score'] >= score_threshold].copy()
    print(f"[PPI] Original edge count: {len (ppi)}, filtered edge count: {len (ppi_filtered)} (score>={score_threshold})")
    return ppi_filtered


def build_adjacency_matrix(ppi_edges, protein_list):
    """
    Construct adjacency matrix

    Parameters:
        ppi_edges: DataFrame, Contains Protein1, Protein2, and combined_store
        protein_list: list, List of protein names (corresponding to feature order)

    return:
        ndarray, Adjacency matrix  (P × P)
    """
    protein_to_idx = {p: i for i, p in enumerate(protein_list)}
    n_proteins = len(protein_list)
    A = np.zeros((n_proteins, n_proteins))
    for _, row in ppi_edges.iterrows():
        p1, p2, score = row['protein1'], row['protein2'], row['combined_score']
        if p1 in protein_to_idx and p2 in protein_to_idx:
            i, j = protein_to_idx[p1], protein_to_idx[p2]
            weight = score
            A[i, j] = weight
            A[j, i] = weight
    print(f"[PPI] adjacency matrix construction completed: {A.shape}, non-zero edges: {np. count_nonzero (A)//2}")
    return A

def compute_graph_laplacian(A, normalized=True):
    """
    Calculate the Turalapras matrix

    Parameters:
        A: ndarray, adjacency matrix
        normalized: bool

    return:
        ndarray, Laplacian matrix L
    """
    A_sparse = csr_matrix(A)
    if normalized:
        L = laplacian(A_sparse, normed=True)
    else:
        L = laplacian(A_sparse, normed=False)

    return L.toarray()