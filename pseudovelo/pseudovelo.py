from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from joblib import Parallel, delayed
import warnings
from typing import Optional, Tuple
from anndata import AnnData
import pandas as pd
import scvelo as scv

def fit_nb_gam_with_center_diff(
    adata,
    pseudotime_key,
    input_layer="X",
    velocity_layer="pseudo_velocity",
    spline_df=10,
    spline_degree=3,
    eps_factor=1e-6,
    verbose=True,
    error_verbose=False,
    reverse=True,
    cluster_key=None,
    n_jobs=1
):
    if input_layer is None:
        X = adata.X.copy()
    else:
        X = adata.layers[input_layer].copy()
    if sparse.issparse(X):
        X = X.toarray()
    if pseudotime_key not in adata.obs:
        raise KeyError(f"'{pseudotime_key}' not found in adata.obs")
    pseudotime_1d = adata.obs[pseudotime_key].values.astype(float)
    if reverse:
        pseudotime_1d = pseudotime_1d.max() - pseudotime_1d
    n_cells = pseudotime_1d.shape[0]
    if n_cells < 2:
        raise ValueError("Need at least 2 cells to fit GAM and compute velocity.")
    n_cells, n_genes = X.shape
    pseudotime_min_all = pseudotime_1d.min()
    pseudotime_max_all = pseudotime_1d.max()
    pseudotime_range_all = pseudotime_max_all - pseudotime_min_all
    
    if pseudotime_range_all <= 0:
        raise ValueError(f"'{pseudotime_key}' pseudotime has zero or negative range; cannot define eps.")
    eps_val = eps_factor * pseudotime_range_all
    if verbose:
        print(f"Start fitting.")
    nb_family = sm.families.NegativeBinomial(
        link=sm.families.links.log()
    )
    def _fit_on_subset(cell_idx, subset_name=None):
        X_sub = X[cell_idx, :]
        pseudotime_sub_1d = pseudotime_1d[cell_idx]
        n_cells_sub = pseudotime_sub_1d.shape[0]
        if n_cells_sub < 2:
            raise ValueError(
                f"Cluster '{subset_name}' has fewer than 2 cells; "
                "cannot fit GAM and compute velocity."
            )
        pseudotime_range_sub = pseudotime_sub_1d.max() - pseudotime_sub_1d.min()
        if pseudotime_range_sub <= 0:
            if verbose:
                if subset_name is None:
                    print(
                        "Subset has zero pseudotime range; use mean expression per gene "
                        "and zero velocity."
                    )
                else:
                    print(
                        f"[cluster={subset_name}] Subset has zero pseudotime range; "
                        "use mean expression per gene and zero velocity."
                    )
            velocity_sub = np.zeros_like(X_sub, dtype=float)
            return velocity_sub
        df_factor = pseudotime_range_sub / pseudotime_range_all
        spline_df_sub = int(np.ceil(spline_df * df_factor))
        min_df = spline_degree + 1
        if spline_df_sub < min_df:
            spline_df_sub = min_df
        pseudotime_sub = pseudotime_sub_1d.reshape(-1, 1)
        if verbose:
            if subset_name is None:
                print(f"Fitting NB GLMGam for {n_genes} genes using n_jobs = {n_jobs}...")
            else:
                print(f"[cluster={subset_name}] Fitting NB GLMGam for {n_genes} genes using n_jobs = {n_jobs}...")
        bs_sub = BSplines(
            pseudotime_sub,
            df=[spline_df_sub],
            degree=[spline_degree],
        )
        exog_sub = np.ones((n_cells_sub, 1))
        velocity_sub = np.zeros_like(X_sub, dtype=float)
        pseudotime_minus_sub = (pseudotime_sub_1d - eps_val).reshape(-1, 1)
        pseudotime_plus_sub = (pseudotime_sub_1d + eps_val).reshape(-1, 1)
        pseudotime_min_sub = pseudotime_sub_1d.min()
        pseudotime_max_sub = pseudotime_sub_1d.max()
        
        pseudotime_minus_sub = np.clip(pseudotime_minus_sub, pseudotime_min_sub, pseudotime_max_sub)
        pseudotime_plus_sub = np.clip(pseudotime_plus_sub, pseudotime_min_sub, pseudotime_max_sub)
        def _fit_single_gene(j):
            y = X_sub[:, j]
            if np.all(y == y[0]):
                vel_j = np.zeros_like(y, dtype=float)
                return j, vel_j
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model = GLMGam(
                        y,
                        exog=exog_sub,
                        smoother=bs_sub,
                        family=nb_family,
                    )
                    res = model.fit()
                pred_expr_minus = res.predict(
                    exog=exog_sub, exog_smooth=pseudotime_minus_sub
                )
                pred_expr_plus = res.predict(
                    exog=exog_sub, exog_smooth=pseudotime_plus_sub
                )
                if (
                    not np.all(np.isfinite(pred_expr_minus))
                    or not np.all(np.isfinite(pred_expr_plus))
                ):
                    raise FloatingPointError("non-finite predicted expression")
                vel_j = (pred_expr_plus - pred_expr_minus) / (2.0 * eps_val)
                if not np.all(np.isfinite(vel_j)):
                    raise FloatingPointError("non-finite velocity")
                return j, vel_j
            except Exception as e:
                if error_verbose:
                    if subset_name is None:
                        print(f"Gene {j} fit failed or unstable: {e}")
                    else:
                        print(f"[cluster={subset_name}] Gene {j} fit failed or unstable: {e}")
                vel_j = np.zeros_like(y, dtype=float)
                return j, vel_j
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_gene)(j) for j in range(n_genes)
        )
        for j, vel_j in results:
            velocity_sub[:, j] = vel_j
        return velocity_sub
    if cluster_key is None:
        if verbose:
            print("Fitting NB GLMGam on all cells.")
        velocity_mat = _fit_on_subset(np.arange(n_cells), subset_name=None)
    else:
        if cluster_key not in adata.obs:
            raise KeyError(f"'{cluster_key}' not found in adata.obs")
        clusters_series = adata.obs[cluster_key]
        if not isinstance(clusters_series.dtype, pd.CategoricalDtype):
            clusters_series = clusters_series.astype("category")
        clusters = clusters_series.values
        unique_clusters = clusters_series.cat.categories
        if verbose:
            print(
                f"Fitting NB GLMGam per cluster, cluster_key = '{cluster_key}', "
                f"n_clusters = {len(unique_clusters)}"
            )
        velocity_mat = np.zeros_like(X, dtype=float)
        for c in unique_clusters:
            cell_idx = np.where(clusters == c)[0]
            if cell_idx.size == 0:
                continue
            vel_sub = _fit_on_subset(cell_idx, subset_name=str(c))
            velocity_mat[cell_idx, :] = vel_sub
    adata.layers[velocity_layer] = velocity_mat
    if verbose:
        print(f"Stored pseudo-velocity in adata.layers['{velocity_layer}'] with shape {velocity_mat.shape}")
    return adata


def velocity_graph(*a, **kw):
    """
    Wrapper function for scvelo's velocity_graph computation.
    """
    import warnings

    warnings.filterwarnings("ignore")

    import builtins

    _real_print = builtins.print
    builtins.print = lambda *_, **__: None
    try:
        return scv.tl.velocity_graph(*a, **kw)
    finally:
        builtins.print = _real_print
        print("computing pseudo-velocity graph\nfinished.")

def velocity_embedding_stream(*a, **kw):
    """
    Wrapper function for scvelo's velocity_embedding_stream visualization.
    """
    import warnings

    warnings.filterwarnings("ignore")

    import builtins

    _real_print = builtins.print
    builtins.print = lambda *_, **__: None
    try:
        return scv.pl.velocity_embedding_stream(*a, **kw)
    finally:
        builtins.print = _real_print
        print("computing pseudo-velocity embedding\nfinished.")

def plot_velocity_projection(
    adata: AnnData,
    vkey: str = "velocity",
    basis: str = "umap",
    xkey: Optional[str] = None,
    color: Optional[str] = None,
    legend_loc: str = "on data",
    title: str = "",
    show: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    size: Optional[int] = None,
    cmap: Optional[str] = None,
    alpha: float = 0.3,
    colorbar: bool = True,
    palette: Optional[str] = None,
    n_jobs: int = 10,
    graph_T: bool = False,
) -> None:
    """
    Computes a velocity graph from the calculated velocity vectors and projects it as a stream plot on the embedding.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with velocities data and embedding coordinates.
    vkey : str, optional
        Layer containing velocities data. Default is 'velocity'.
    basis : str, optional
        Embedding basis to use (e.g., 'umap', 'tsne'). Default is 'umap'.
    xkey : str, optional
        Layer to use as expression data.
        If None, uses adata.X converted to dense array. Default is None.
    color : str, optional
        Column name in obs for coloring points. If None, no coloring is applied.
        Default is None.
    legend_loc : str, optional
        Location of legend. Default is 'on data'.
    title : str, optional
        Plot title. Default is empty string.
    show : bool, optional
        Whether to display the plot. Default is True.
    figsize : tuple of int, optional
        Figure size (width, height) in inches. Default is (8, 6).
    size : int, optional
        Size of points in scatter plot. If None, uses default. Default is None.
    cmap : str, optional
        Colormap for coloring. If None, uses default. Default is None.
    alpha : float, optional
        Transparency of points. Default is 0.3.
    colorbar : bool, optional
        Whether to show colorbar. Default is True.
    palette : str, optional
        Color palette for categorical coloring. If None, uses default.
        Default is None.
    n_jobs : int, optional
        Number of jobs for parallel computation. Default is 10.
    graph_T : bool, optional
        If True, transpose the adjacency matrix. Default is False.

    Returns
    -------
    None
        Shows the velocity projection plot.

    Raises
    ------
    ValueError
        If required layers or embeddings are not found.
    """

    if vkey not in adata.layers:
        raise ValueError(f"Layer '{vkey}' not found in adata.layers")

    embedding_key = f"X_{basis}" if not basis.startswith("X_") else basis
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding coordinates '{embedding_key}' not found in adata.obsm"
        )

    if xkey is None:
        xkey = "X"
        if hasattr(adata.X, "toarray"):
            adata.layers["X"] = adata.X.toarray()
    elif xkey not in adata.layers:
        raise ValueError(f"Layer '{xkey}' not found in adata.layers")

    if color is not None and color not in adata.obs.columns:
        raise ValueError(f"Column '{color}' not found in adata.obs")

    velocity_graph(adata, vkey=vkey, xkey=xkey, n_jobs=n_jobs)

    graph_key = f"{vkey}_graph"
    if graph_key not in adata.uns:
        raise KeyError(graph_key)
    if graph_T is True:
        adata.uns[graph_key] = adata.uns[graph_key].T

    velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=vkey,
        color=color,
        legend_loc=legend_loc,
        title=title,
        show=show,
        figsize=figsize,
        size=size,
        cmap=cmap,
        alpha=alpha,
        colorbar=colorbar,
        palette=palette,
    )