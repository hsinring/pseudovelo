## PseudoVelo: Inferring gene expression derivatives along pseudotime as pseudo-velocity.

**PseudoVelo** is a computational framework that infers pseudo-velocity directly from pseudotime. PseudoVelo utilizes Generalized Additive Models (GAMs) to fit gene expression dynamics and employs a central difference approximation to calculate the instantaneous rate of change for each gene. This straightforward yet effective approach directly yields a pseudo-velocity matrix that is fully compatible with existing downstream single-cell workflows.

### Installation
You can install PseudoVelo directly from GitHub using pip. Since it requires Python 3.9 or higher, please ensure your environment meets this requirement:
```bash
pip install git+https://github.com/hsinring/pseudovelo.git
```

### Quick Start
Below is a basic workflow demonstrating how to use PseudoVelo after computing pseudotime (e.g., using DPT).
```bash
import scanpy as sc
import pseudovelo as pv

# 1. Load your AnnData object
adata = sc.read_h5ad("your_data.h5ad")

# (Optional) Calculate pseudotime here if not already computed
# sc.tl.dpt(adata) 

# 2. Prepare the data matrix
# Ensure the input data is in a dense format within adata.layers
# Note: The layer doesn't have to be "X", it can also be "Ms" or other layers depending on your workflow.
adata.layers["X"] = adata.X.toarray()

# 3. Fit GAM and calculate pseudo_velocity
# This step fits the Negative Binomial GAMs and computes the central difference
pv.fit_nb_gam_with_center_diff(
    adata,
    J_key="dpt_pseudotime",    # Key for pseudotime in adata.obs
    n_jobs=10,                 # Number of parallel jobs
    cluster_key="lineages",    # Key for cell lineages/clusters in adata.obs (default: None for unbranched)
    input_layer="X",           # Layer to use for fitting
    reverse=False
)

# 4. Plot pseudo-velocity projection
# Visualize the inferred velocity streamlines or arrows
pv.plot_velocity_projection(
    adata,
    xkey="X",                  # Basis/Embedding key for projection (e.g., 'X_umap')
    vkey="pseudo_velocity",      # Key where the calculated pseudo_velocity is stored
    color="lineages"           # Color cells by lineage/cluster
)

```



