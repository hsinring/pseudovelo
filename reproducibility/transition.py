import warnings

import numpy as np
import scipy.sparse as sp
from scipy import sparse
from scipy.sparse import coo_matrix,issparse,csr_matrix
from sklearn.metrics.pairwise import cosine_similarity



def get_l2_norm(x, axis: int = 1):

    if sparse.issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    elif x.ndim == 1:
        return np.sqrt(np.einsum("i, i -> ", x, x))
    elif axis == 0:
        return np.sqrt(np.einsum("ij, ij -> j", x, x))
    elif axis == 1:
        return np.sqrt(np.einsum("ij, ij -> i", x, x))

def normalize(X):
    """TODO."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if issparse(X):
            return X.multiply(csr_matrix(1.0 / np.abs(X).sum(1)))
        else:
            return X / X.sum(1)



def transition_matrix(
    adata,
    vkey="pseudo_velocity",
    backward=False,
    self_transitions=True,
    scale=10,
    use_negative_cosines=False,
):
    if f"{vkey}_graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.velocity_graph` first to compute cosine correlations."
        )

    graph_neg = None

    graph = csr_matrix(adata.uns[f"{vkey}_graph"]).copy()
    if f"{vkey}_graph_neg" in adata.uns.keys():
        graph_neg = adata.uns[f"{vkey}_graph_neg"]

    if self_transitions:
        confidence = graph.max(1).toarray().flatten()
        ub = np.percentile(confidence, 98)
        self_prob = np.clip(ub - confidence, 0, 1)
        graph.setdiag(self_prob)

    T = np.expm1(graph * scale) 
    
    if use_negative_cosines:
        T -= np.expm1(-graph_neg * scale)
    else:
        T += np.expm1(graph_neg * scale)
        T.data += 1

    if backward:
        T = T.T
    
    T.eliminate_zeros()
    T = normalize(T)

    return T


def Expected_velos(X_emb, T, l2_norm=False):
    
    densify = X_emb.shape[0] < 1e4
    TA = T.toarray() if densify else None

    VS = np.zeros(X_emb.shape)

    for obs_id in range(X_emb.shape[0]):
        indices = T[obs_id].indices
        dX = X_emb[indices] - X_emb[obs_id, None]
        probs = TA[obs_id, indices] if densify else T[obs_id].data

        if l2_norm:
            dX_normal = get_l2_norm(dX).reshape(-1,1)
            dX_normal[dX_normal == 0] = 1
            dX = dX / dX_normal

        VS[obs_id] = probs.dot(dX) - probs.mean() * dX.sum(0)

    return VS


def evaluate_transition_accuracy(
    adata,
    cluster_key: str,
    lineage_order: list,
    vkey: str = "pseudo_velocity",
    T=None,
):
    if T is None:
        T = transition_matrix(
            adata=adata,
            vkey=vkey,
            backward=False,
            self_transitions=True,
            scale=10,
            use_negative_cosines=False,
        )
    
    T = csr_matrix(T)
    if cluster_key not in adata.obs:
        raise ValueError(f"'{cluster_key}' not found in adata.obs")
    
    clusters = adata.obs[cluster_key].astype(str).values
    
    if len(lineage_order) > 0 and not isinstance(lineage_order[0], (list, tuple)):
        lineage_order = [lineage_order]
        
    correct_edges = set()
    wrong_edges = set()
    
    for lineage in lineage_order:
        for i in range(len(lineage) - 1):
            u = str(lineage[i])
            v = str(lineage[i+1])
            correct_edges.add((u, v))
            wrong_edges.add((v, u))
            
    if not correct_edges:
        return 0.0
            
    unique_clusters = list(set([c for edge in correct_edges for c in edge] + 
                               [c for edge in wrong_edges for c in edge]))
    cluster_to_id = {c: i for i, c in enumerate(unique_clusters)}
    N = len(unique_clusters)
    
    correct_edge_ids = np.array([cluster_to_id[u] * N + cluster_to_id[v] for u, v in correct_edges])
    wrong_edge_ids = np.array([cluster_to_id[u] * N + cluster_to_id[v] for u, v in wrong_edges])
    
    cell_orders = np.array([cluster_to_id.get(c, -1) for c in clusters])
    
    T = coo_matrix(T)
    rows = T.row
    cols = T.col
    data = T.data
    
    start_orders = cell_orders[rows]
    end_orders = cell_orders[cols]
    
    valid_mask = (start_orders != -1) & (end_orders != -1)
    
    valid_starts = start_orders[valid_mask]
    valid_ends = end_orders[valid_mask]
    valid_data = data[valid_mask]
    
    edge_ids = valid_starts * N + valid_ends
    
    correct_mask = np.isin(edge_ids, correct_edge_ids)
    wrong_mask = np.isin(edge_ids, wrong_edge_ids)
    
    correct_weight = np.sum(valid_data[correct_mask])
    wrong_weight = np.sum(valid_data[wrong_mask])
    
    total_weight = correct_weight + wrong_weight

    if total_weight == 0:
        return 0.0
        
    accuracy_ratio = correct_weight / total_weight
    return accuracy_ratio



def projection_consistency(adata, embedding_key='X_umap', vkey='pseudo_velocity',
                           conn_key='connectivities', obsm_key='expected_velos', l2_norm=False, T_velo=None):
    
    X_emb = adata.obsm[embedding_key]
    T_connect = adata.obsp[conn_key]
    if not sp.isspmatrix_csr(T_connect):
        T_connect = T_connect.tocsr()
    if T_velo is None:
        T_velo = transition_matrix(
            adata=adata,
            vkey=vkey,
            backward=False,
            self_transitions=True,
            scale=10,
            use_negative_cosines=False,
        )
    
    if not sp.isspmatrix_csr(T_velo):
        T_velo = T_velo.tocsr()
    VS = Expected_velos(X_emb, T_velo, l2_norm=l2_norm)
    adata.obsm[obsm_key] = VS
    n_cells = X_emb.shape[0]
    consistency = np.zeros(n_cells)
    
    for obs_id in range(n_cells):

        neighbor_indices = T_connect[obs_id].indices
        if len(neighbor_indices) == 0:
            consistency[obs_id] = np.nan
            continue
            
        v_ref = VS[obs_id].reshape(1, -1)
        v_neighbors = VS[neighbor_indices]
        
        cos_sims = cosine_similarity(v_ref, v_neighbors).flatten()
        consistency[obs_id] = cos_sims.mean()
        
    return consistency

def cross_projection_consistency(adata1, adata2=None, embedding_key='X_umap', vkey='pseudo_velocity', 
                                 l2_norm=False, T_velo1=None, T_velo2=None):
    
    X_emb = adata1.obsm[embedding_key]
    if T_velo1 is None:
        T_velo1 = transition_matrix(adata1, vkey=vkey)
    VS1 = Expected_velos(X_emb, T_velo1, l2_norm=l2_norm)
    if T_velo2 is None:
        T_velo2 = transition_matrix(adata2, vkey=vkey)
    VS2 = Expected_velos(X_emb, T_velo2, l2_norm=l2_norm)
    dot_product = np.einsum('ij,ij->i', VS1, VS2)
    norm1 = get_l2_norm(VS1, axis=1)
    norm2 = get_l2_norm(VS2, axis=1)
    consistency = np.full(X_emb.shape[0], np.nan)
    denom = norm1 * norm2
    valid_mask = denom > 0
    consistency[valid_mask] = dot_product[valid_mask] / denom[valid_mask]
    
    return consistency


