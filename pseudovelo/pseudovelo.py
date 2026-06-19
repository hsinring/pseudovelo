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
    J_key,
    input_layer="X",
    output_layer="gam_nb_mu",
    dmu_layer="gam_nb_dmu_dJ",
    k_mu=10,
    degree_mu=3,
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

    if J_key not in adata.obs:
        raise KeyError(f"{J_key} not found in adata.obs")

    J_1d = adata.obs[J_key].values.astype(float)

    if reverse:
        J_1d = J_1d.max() - J_1d

    n_cells = J_1d.shape[0]
    if n_cells < 2:
        raise ValueError("Need at least 2 cells to fit GAM and compute derivatives.")

    n_cells, n_genes = X.shape

    J_min_global = J_1d.min()
    J_max_global = J_1d.max()
    J_range_global = J_max_global - J_min_global
    if J_range_global <= 0:
        raise ValueError("Global J has zero or negative range; cannot define eps.")

    eps_global = eps_factor * J_range_global

    if verbose:
        print(
            f"[global] J range = {J_range_global:.6g}, "
            f"eps = {eps_global:.6g} (eps_factor = {eps_factor})"
        )

    nb_family = sm.families.NegativeBinomial(
        link=sm.families.links.log()
    )

    def _fit_on_subset(cell_idx, subset_name=None):
        X_sub = X[cell_idx, :]
        J_sub_1d = J_1d[cell_idx]

        n_cells_sub = J_sub_1d.shape[0]
        if n_cells_sub < 2:
            raise ValueError(
                f"Cluster '{subset_name}' has fewer than 2 cells; "
                "cannot fit GAM and compute derivatives."
            )

        J_range_sub = J_sub_1d.max() - J_sub_1d.min()
        if J_range_sub <= 0:
            if verbose:
                if subset_name is None:
                    print(
                        "Subset has zero J range; use mean expression per gene "
                        "and zero derivative."
                    )
                else:
                    print(
                        f"[cluster={subset_name}] Subset has zero J range; "
                        "use mean expression per gene and zero derivative."
                    )
            mu_hat_sub = np.zeros_like(X_sub, dtype=float)
            dmu_dJ_sub = np.zeros_like(X_sub, dtype=float)
            for j in range(n_genes):
                y = X_sub[:, j]
                const_val = y.mean()
                mu_hat_sub[:, j] = const_val
                dmu_dJ_sub[:, j] = 0.0
            return mu_hat_sub, dmu_dJ_sub

        k_mu_factor = J_range_sub / J_range_global
        k_mu_sub = int(np.ceil(k_mu * k_mu_factor))
        min_df = degree_mu + 1
        if k_mu_sub < min_df:
            k_mu_sub = min_df

        J_sub = J_sub_1d.reshape(-1, 1)

        if verbose:
            if subset_name is None:
                print(
                    f"Fitting NB GLMGam for {n_genes} genes (center difference), "
                    f"k_mu_sub = {k_mu_sub} (factor = {k_mu_factor:.3g}), "
                    f"n_jobs = {n_jobs}."
                )
            else:
                print(
                    f"[cluster={subset_name}] Fitting NB GLMGam for {n_genes} genes "
                    f"(center difference), k_mu_sub = {k_mu_sub} "
                    f"(factor = {k_mu_factor:.3g}), n_jobs = {n_jobs}."
                )

        bs_sub = BSplines(
            J_sub,
            df=[k_mu_sub],
            degree=[degree_mu],
        )

        exog_sub = np.ones((n_cells_sub, 1))

        mu_hat_sub = np.zeros_like(X_sub, dtype=float)
        dmu_dJ_sub = np.zeros_like(X_sub, dtype=float)

        J_minus_sub = (J_sub_1d - eps_global).reshape(-1, 1)
        J_plus_sub = (J_sub_1d + eps_global).reshape(-1, 1)

        J_min_sub, J_max_sub = J_sub_1d.min(), J_sub_1d.max()
        J_minus_sub = np.clip(J_minus_sub, J_min_sub, J_max_sub)
        J_plus_sub = np.clip(J_plus_sub, J_min_sub, J_max_sub)

        def _fit_single_gene(j):
            y = X_sub[:, j]

            if np.all(y == y[0]):
                const_val = y.mean()
                mu_j = np.full_like(y, const_val, dtype=float)
                dmu_j = np.zeros_like(y, dtype=float)
                return j, mu_j, dmu_j

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

                mu_J_sub = res.predict(exog=exog_sub, exog_smooth=J_sub)
                mu_J_minus_sub = res.predict(
                    exog=exog_sub, exog_smooth=J_minus_sub
                )
                mu_J_plus_sub = res.predict(
                    exog=exog_sub, exog_smooth=J_plus_sub
                )

                if (
                    not np.all(np.isfinite(mu_J_sub))
                    or not np.all(np.isfinite(mu_J_minus_sub))
                    or not np.all(np.isfinite(mu_J_plus_sub))
                ):
                    raise FloatingPointError("non-finite mu in prediction")

                dmu_j = (mu_J_plus_sub - mu_J_minus_sub) / (2.0 * eps_global)

                if not np.all(np.isfinite(dmu_j)):
                    raise FloatingPointError("non-finite derivative")

                return j, mu_J_sub, dmu_j

            except Exception as e:
                if error_verbose:
                    if subset_name is None:
                        print(f"Gene {j} fit failed or unstable: {e}")
                    else:
                        print(f"[cluster={subset_name}] Gene {j} fit failed or unstable: {e}")
                const_val = y.mean()
                mu_j = np.full_like(y, const_val, dtype=float)
                dmu_j = np.zeros_like(y, dtype=float)
                return j, mu_j, dmu_j

        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_gene)(j) for j in range(n_genes)
        )

        for j, mu_j, dmu_j in results:
            mu_hat_sub[:, j] = mu_j
            dmu_dJ_sub[:, j] = dmu_j

        return mu_hat_sub, dmu_dJ_sub

    if cluster_key is None:
        if verbose:
            print("Fitting NB GLMGam on all cells (no clustering).")
        mu_hat, dmu_dJ = _fit_on_subset(np.arange(n_cells), subset_name=None)

    else:
        if cluster_key not in adata.obs:
            raise KeyError(f"{cluster_key} not found in adata.obs")

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

        mu_hat = np.zeros_like(X, dtype=float)
        dmu_dJ = np.zeros_like(X, dtype=float)

        for c in unique_clusters:
            cell_idx = np.where(clusters == c)[0]
            if cell_idx.size == 0:
                continue
            mu_sub, dmu_sub = _fit_on_subset(cell_idx, subset_name=str(c))
            mu_hat[cell_idx, :] = mu_sub
            dmu_dJ[cell_idx, :] = dmu_sub

    adata.layers[output_layer] = mu_hat
    adata.layers[dmu_layer] = dmu_dJ

    if verbose:
        print(f"Stored μ in adata.layers['{output_layer}'] with shape {mu_hat.shape}")
        print(f"Stored dμ/dJ in adata.layers['{dmu_layer}'] with shape {dmu_dJ.shape}")

    return adata

def fit_nb_gam(
    adata,
    J_key,
    input_layer="X",
    output_layer="gam_mu",
    k_mu=10,
    degree_mu=3,
    verbose=True,
    error_verbose=False,
    cluster_key=None,
    n_jobs=1,
    reverse=True,
    gene_list=None,
):
    if input_layer is None:
        X = adata.X.copy()
    else:
        X = adata.layers[input_layer].copy()

    if sparse.issparse(X):
        X = X.toarray()

    if J_key not in adata.obs:
        raise KeyError(f"{J_key} not found in adata.obs")

    J_1d = adata.obs[J_key].values.astype(float)
    if reverse:
        J_1d = J_1d.max() - J_1d

    n_cells = J_1d.shape[0]
    if n_cells < 2:
        raise ValueError("Need at least 2 cells to fit GAM.")

    n_cells, n_genes = X.shape

    if gene_list is None:
        gene_indices = np.arange(n_genes, dtype=int)
        n_genes_fit = n_genes
        if verbose:
            print(f"Fitting NB GLMGam for all {n_genes_fit} genes.")
    else:
        if isinstance(gene_list, (str, bytes)):
            gene_list = [gene_list]
        gene_list = list(gene_list)

        missing = [g for g in gene_list if g not in adata.var_names]
        if len(missing) > 0:
            raise KeyError(f"These genes not found in adata.var_names: {missing}")

        gene_indices = np.array(
            [int(np.where(adata.var_names == g)[0][0]) for g in gene_list],
            dtype=int,
        )
        n_genes_fit = len(gene_indices)
        if verbose:
            print(f"Fitting NB GLMGam for {n_genes_fit} genes in gene_list.")

    J_min_global = J_1d.min()
    J_max_global = J_1d.max()
    J_range_global = J_max_global - J_min_global
    if J_range_global <= 0:
        if verbose:
            print("Global J has zero range. Using mean expression per gene.")
        mu_hat_full = np.zeros((n_cells, n_genes), dtype=float)
        for j in gene_indices:
            mu_hat_full[:, j] = X[:, j].mean()
        adata.layers[output_layer] = mu_hat_full
        if verbose:
            print(f"Stored μ in adata.layers['{output_layer}'] with shape {mu_hat_full.shape}")
        return adata

    nb_family = sm.families.NegativeBinomial(link=sm.families.links.log())

    def _fit_on_subset(cell_idx, subset_name=None):
        X_sub = X[cell_idx, :]
        J_sub_1d = J_1d[cell_idx]

        n_cells_sub = J_sub_1d.shape[0]
        if n_cells_sub < 2:
            raise ValueError(
                f"Cluster '{subset_name}' has fewer than 2 cells; cannot fit GAM."
            )

        J_range_sub = J_sub_1d.max() - J_sub_1d.min()
        if J_range_sub <= 0:
            if verbose:
                if subset_name is None:
                    print("Subset has zero J range; use mean expression per gene.")
                else:
                    print(
                        f"[cluster={subset_name}] Subset has zero J range; "
                        "use mean expression per gene."
                    )
            mu_hat_sub = np.zeros((n_cells_sub, n_genes_fit), dtype=float)
            for k, j in enumerate(gene_indices):
                y = X_sub[:, j]
                mu_hat_sub[:, k] = y.mean()
            return mu_hat_sub

        k_mu_factor = J_range_sub / J_range_global
        k_mu_sub = int(np.ceil(k_mu * k_mu_factor))
        min_df = degree_mu + 1
        if k_mu_sub < min_df:
            k_mu_sub = min_df

        J_sub = J_sub_1d.reshape(-1, 1)

        if verbose:
            if subset_name is None:
                print(
                    f"Fitting NB GLMGam for {n_genes_fit} genes, "
                    f"k_mu_sub = {k_mu_sub} (factor = {k_mu_factor:.3g}), "
                    f"n_jobs = {n_jobs}."
                )
            else:
                print(
                    f"[cluster={subset_name}] Fitting NB GLMGam for {n_genes_fit} genes, "
                    f"k_mu_sub = {k_mu_sub} (factor = {k_mu_factor:.3g}), "
                    f"n_jobs = {n_jobs}."
                )

        bs_sub = BSplines(
            J_sub,
            df=[k_mu_sub],
            degree=[degree_mu],
        )

        exog_sub = np.ones((n_cells_sub, 1))
        mu_hat_sub = np.zeros((n_cells_sub, n_genes_fit), dtype=float)

        def _fit_single_gene(idx_and_j):
            k, j = idx_and_j
            y = X_sub[:, j]

            if np.all(y == y[0]):
                const_val = y.mean()
                mu_j = np.full_like(y, const_val, dtype=float)
                return k, mu_j

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

                mu_J_sub = res.predict(exog=exog_sub, exog_smooth=J_sub)

                if not np.all(np.isfinite(mu_J_sub)):
                    raise FloatingPointError("non-finite mu in prediction")

                return k, mu_J_sub

            except Exception as e:
                if error_verbose:
                    if subset_name is None:
                        print(f"Gene index {j} fit failed or unstable: {e}")
                    else:
                        print(
                            f"[cluster={subset_name}] Gene index {j} fit failed or unstable: {e}"
                        )
                const_val = y.mean()
                mu_j = np.full_like(y, const_val, dtype=float)
                return k, mu_j

        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_single_gene)(ij) for ij in enumerate(gene_indices)
        )

        for k, mu_j in results:
            mu_hat_sub[:, k] = mu_j

        return mu_hat_sub

    mu_hat_full = np.zeros((n_cells, n_genes), dtype=float)

    if cluster_key is None:
        if verbose:
            print("Fitting NB GLMGam on all cells (no clustering).")
        mu_sub = _fit_on_subset(np.arange(n_cells), subset_name=None)
        for k, j in enumerate(gene_indices):
            mu_hat_full[:, j] = mu_sub[:, k]
    else:
        if cluster_key not in adata.obs:
            raise KeyError(f"{cluster_key} not found in adata.obs")

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

        for c in unique_clusters:
            cell_idx = np.where(clusters == c)[0]
            if cell_idx.size == 0:
                continue
            mu_sub = _fit_on_subset(cell_idx, subset_name=str(c))
            for k, j in enumerate(gene_indices):
                mu_hat_full[cell_idx, j] = mu_sub[:, k]

    adata.layers[output_layer] = mu_hat_full

    if verbose:
        print(f"Stored μ in adata.layers['{output_layer}'] with shape {mu_hat_full.shape}")

    return adata

def plot_gam_fit(
    adata,
    J_key,
    input_layer="X",
    mu_layer="gam_mu",
    n_genes_to_plot=5,
    random_state=None,
    figsize=(15, 10),
    cluster_key=None,
    gene_list=None,
    xlabel="Pseudotime",
    ylabel="Gene expression",
    fontsize=12
):
    if J_key not in adata.obs:
        raise KeyError(f"{J_key} not found in adata.obs")
    J = adata.obs[J_key].values.astype(float)

    if input_layer is None:
        X = adata.X
    else:
        X = adata.layers[input_layer]

    if sparse.issparse(X):
        X = X.toarray()

    if mu_layer not in adata.layers:
        raise KeyError(
            f"{mu_layer} not found in adata.layers (did you run fit_nb_gam_by_gene?)"
        )
    MU = adata.layers[mu_layer]
    if sparse.issparse(MU):
        MU = MU.toarray()

    n_cells, n_genes = X.shape

    if gene_list is not None:
        gene_list = list(gene_list)
        missing_genes = [g for g in gene_list if g not in adata.var_names]
        if missing_genes:
            raise KeyError(f"Genes not found in adata.var_names: {missing_genes}")
        gene_indices = np.array(adata.var_names.get_indexer(gene_list))
        n_genes_to_plot = len(gene_indices)
        gene_names = np.array(gene_list)
    else:
        rng = np.random.default_rng(random_state)
        if n_genes_to_plot > n_genes:
            n_genes_to_plot = n_genes
        gene_indices = rng.choice(n_genes, size=n_genes_to_plot, replace=False)
        gene_names = adata.var_names[gene_indices]

    if cluster_key is not None:
        if cluster_key not in adata.obs:
            raise KeyError(f"{cluster_key} not found in adata.obs")
        if f"{cluster_key}_colors" not in adata.uns:
            raise KeyError(f"{cluster_key}_colors not found in adata.uns")

        clusters_series = adata.obs[cluster_key].astype("category")
        clusters = clusters_series.values
        unique_clusters = clusters_series.cat.categories

        raw_colors = adata.uns[f"{cluster_key}_colors"]
        if len(raw_colors) < len(unique_clusters):
            raise ValueError(
                f"{cluster_key}_colors has fewer colors ({len(raw_colors)}) "
                f"than clusters ({len(unique_clusters)})"
            )

        cluster_colors = {c: raw_colors[i] for i, c in enumerate(unique_clusters)}
    else:
        clusters = None
        unique_clusters = None
        cluster_colors = None

    nrows = int(np.ceil(n_genes_to_plot / 2))
    ncols = 2 if n_genes_to_plot > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    if xlabel is None:
        xlabel = f"{J_key} (aligned)"

    for ax, idx, gname in zip(axes, gene_indices, gene_names):
        y = X[:, idx]
        mu = MU[:, idx]

        if cluster_key is None:
            J_aligned = J - J.min()
            order_global = np.argsort(J_aligned)
            J_sorted_global = J_aligned[order_global]
            mu_sorted = mu[order_global]

            ax.scatter(J_aligned, y, color="tab:red", s=3, alpha=0.1)
            ax.plot(
                J_sorted_global,
                mu_sorted,
                color="tab:red",
                linewidth=2,
                label="Fitted μ",
                alpha=0.8,
            )
        else:
            for c in unique_clusters:
                color = cluster_colors[c]
                cell_idx = np.where(clusters == c)[0]
                if cell_idx.size == 0:
                    continue

                J_c = J[cell_idx]
                mu_c = mu[cell_idx]
                y_c = y[cell_idx]

                order_c = np.argsort(J_c)
                J_c_sorted = J_c[order_c]
                mu_c_sorted = mu_c[order_c]

                ax.scatter(J_c, y_c, color=color, s=3, alpha=0.1)
                ax.plot(
                    J_c_sorted,
                    mu_c_sorted,
                    color=color,
                    linewidth=2,
                    label=f"{c} fitted μ",
                    alpha=0.8,
                )

        ax.set_title(str(gname), fontsize=fontsize)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize-2)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        origin_x = xlim[0] + 0.05 * (xlim[1] - xlim[0])
        origin_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])

        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position(("data", origin_y - 0.03 * (ylim[1] - ylim[0])))
        ax.spines["left"].set_position(("data", origin_x - 0.03 * (xlim[1] - xlim[0])))

    for k in range(n_genes_to_plot, len(axes)):
        fig.delaxes(axes[k])

    fig.tight_layout()
    plt.show()

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
        print("computing velocity graph\nfinished.")

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
        print("computing velocity embedding\nfinished.")

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