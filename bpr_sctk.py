# =============================================================================
# bpr_sctoolkit.py
# =============================================================================
"""
Brett's single-cell/multimodal utilities for Scanpy/MuData workflows.

This file consolidates all helpers and core functions you provided into ONE
drop-in module with detailed, NumPy-style docstrings and example usage.

No function names or behaviors have been changed. Only documentation, imports,
and light organization comments were added for clarity.

---------------------------------------------------------------------------
MASTER INDEX (search these anchors)
---------------------------------------------------------------------------
[UTILS]
- clear_gpu_memory

[QC & METRICS]
- is_outlier
- plot_qc_summary
- build_marker_qc_masks_strict
- show_marker_qc

[DOUBLETS]
- run_scrublet_scores

[CLEANUP / STRIPPING]
- remove_elements

[DIMRED / CLUSTERING / TOPOLOGY]
- umap_optimizer
- leiden_optimizer
- generate_processed_PAGA_UMAP
- run_umap_leiden_grid  (RAPIDS-aware UMAP/Leiden grid)

[ADT / PROTEIN PROCESSING]
- clr_prot
- run_gmm_thresholding   (marker-first pooled-BIC + per-sample GMM)

[VISUALIZATION]
- plot_umap_by_genes_and_subtypes
- histogram_kde_facet_vertical_multi
- paged_plot
"""

# =============================================================================
# Imports (unified)
# =============================================================================
from __future__ import annotations

import os
import re
import gc
import inspect
from collections import OrderedDict
from typing import Optional, Union, Sequence, Tuple, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

from IPython.display import display, clear_output
import ipywidgets as widgets

import scanpy as sc
import anndata as ad

from scipy import sparse
from scipy.stats import median_abs_deviation
from sklearn.mixture import GaussianMixture

# Optional: GPU (cupy) — will be used if available.
try:
    import cupy as cp  # noqa: F401
except Exception:
    cp = None  # keep as None so utils can no-op if CuPy isn't present


# =============================================================================
# [UTILS]
# =============================================================================
def clear_gpu_memory():
    """
    Best-effort GPU memory cleanup.

    Tries to:
    1) Run Python garbage collection
    2) If CuPy is available, free its default memory pool

    Notes
    -----
    - If CuPy isn't installed/available, this function simply runs `gc.collect()`.
    - No return value; purely side-effects.

    Examples
    --------
    >>> clear_gpu_memory()
    """
    gc.collect()
    if cp is not None:
        try:
            cp._default_memory_pool.free_all_blocks()
        except Exception:
            # If the pool or cp fails for any reason, we silently continue
            pass


# =============================================================================
# [QC & METRICS]
# =============================================================================
def is_outlier(adata, metric: str, nmads: int):
    """
    Flag outliers in `adata.obs[metric]` using a Median Absolute Deviation (MAD) rule.

    Parameters
    ----------
    adata : AnnData
        AnnData object with the relevant metric in `.obs`.
    metric : str
        Column in `adata.obs` whose values will be tested.
    nmads : int
        Threshold in MADs from the median. Values outside
        [median - nmads*MAD, median + nmads*MAD] are flagged as outliers.

    Returns
    -------
    pandas.Series (bool)
        Boolean mask of length `adata.n_obs` where True marks an outlier.

    Examples
    --------
    >>> mask = is_outlier(adata, metric="pct_counts_mt", nmads=3)
    >>> adata.obs["mt_outlier"] = mask
    """
    M = adata.obs[metric]
    med = np.median(M)
    mad = median_abs_deviation(M)
    outlier = (M < med - nmads * mad) | (med + nmads * mad < M)
    return outlier


def plot_qc_summary(sample_id, rna, prot, stage="Before"):
    """
    Plot quick QC summaries for RNA and protein modalities.

    Parameters
    ----------
    sample_id : str
        An identifier used in the figure title.
    rna : AnnData
        RNA AnnData with required columns in `.obs`:
        - "n_genes_by_counts"
        - "pct_counts_mt"
        Optionally "qc_filter_pass" (bool) to color distributions pre-filter.
    prot : AnnData
        Protein AnnData with required columns in `.obs`:
        - "n_genes_by_counts"
        - "total_counts"
        Optionally "pct_counts_in_top_20_genes".
    stage : {"Before","After"}, default: "Before"
        Controls which histograms are drawn for RNA.

    Returns
    -------
    None
        Displays a Matplotlib figure.

    Examples
    --------
    >>> plot_qc_summary("S1", rna, prot, stage="Before")
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"QC Metrics ({stage}) - {sample_id}", fontsize=16)

    # --- RNA plots ---
    if stage == "Before":
        sns.histplot(
            data=rna.obs, x="n_genes_by_counts", hue="qc_filter_pass",
            palette={True: "green", False: "red"},
            ax=axes[0, 0], bins=50, element="step", stat="density"
        )
        axes[0, 0].set_title("RNA: n_genes_by_counts (Pre)")

        sns.histplot(
            data=rna.obs, x="pct_counts_mt", hue="qc_filter_pass",
            palette={True: "green", False: "red"},
            ax=axes[0, 1], bins=50, element="step", stat="density"
        )
        axes[0, 1].set_title("RNA: pct_counts_mt (Pre)")
    else:
        sns.histplot(rna.obs["n_genes_by_counts"], color="green", ax=axes[0, 0], bins=50, element="step", stat="density")
        axes[0, 0].set_title("RNA: n_genes_by_counts (Post)")

        sns.histplot(rna.obs["pct_counts_mt"], color="green", ax=axes[0, 1], bins=50, element="step", stat="density")
        axes[0, 1].set_title("RNA: pct_counts_mt (Post)")

    axes[0, 0].set_xlabel("n_genes_by_counts")
    axes[0, 1].set_xlabel("pct_counts_mt")

    # Leave 3rd column in RNA row empty
    axes[0, 2].axis("off")

    # --- Protein plots ---
    sns.histplot(prot.obs['n_genes_by_counts'], bins=50, ax=axes[1, 0], kde=True, color="blue")
    axes[1, 0].set_title("Protein: n_genes_by_counts")
    axes[1, 0].set_xlabel("n_genes_by_counts")

    sns.histplot(prot.obs['total_counts'], bins=50, ax=axes[1, 1], kde=True, color="blue")
    axes[1, 1].set_title("Protein: total_counts")
    axes[1, 1].set_xlabel("total_counts")

    if 'pct_counts_in_top_20_genes' in prot.obs.columns:
        sns.histplot(prot.obs['pct_counts_in_top_20_genes'], bins=50, ax=axes[1, 2], kde=True, color="blue")
        axes[1, 2].set_title("Protein: pct_counts_in_top_20_genes")
        axes[1, 2].set_xlabel("pct_counts_in_top_20_genes")
    else:
        axes[1, 2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def build_marker_qc_masks_strict(
    adata_prot,
    layer: Optional[str] = None,
    zero_thresh: float = 0.98,
    var_eps: float = 1e-10,
    min_nonzero_frac: float = 0.02,
    abs_range_eps: float = 1e-6,
    rel_range_eps: float = 1e-3,
    treat_nan_as_zero: bool = False,
    store: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Strict QC for protein markers (sparse- and NaN-robust).

    A marker fails if ANY of:
      A) variance <= var_eps
      B) zero fraction >= zero_thresh
      C) nonzero fraction < min_nonzero_frac
      D) ≤ 1 unique finite value
      E) robust range (p99 - p1) < abs_range_eps
      F) robust range (p99 - p1) < rel_range_eps * (median + 1)

    Parameters
    ----------
    adata_prot : AnnData
        Protein AnnData object (typically `mdata.mod['prot']`).
    layer : str or None, default: None
        Use a numeric layer for QC if provided; otherwise `.X`.
    zero_thresh : float, default: 0.98
        Maximum allowed fraction of zeros; values >= this mark as fail.
    var_eps : float, default: 1e-10
        Minimum allowed variance.
    min_nonzero_frac : float, default: 0.02
        Require at least this fraction of non-zero entries.
    abs_range_eps : float, default: 1e-6
        Absolute (p99 - p1) lower bound.
    rel_range_eps : float, default: 1e-3
        Relative lower bound (p99 - p1) < rel_range_eps * (median + 1).
    treat_nan_as_zero : bool, default: False
        If True, NaNs are treated as zeros; else they are excluded from stats.
    store : bool, default: True
        If True, store QC metrics/flags into `adata_prot.var[...]`.

    Returns
    -------
    passed_mask : np.ndarray (bool)
    failed_mask : np.ndarray (bool)
    passed_markers : list[str]
    failed_markers : list[str]

    Examples
    --------
    >>> pass_mask, fail_mask, passed, failed = build_marker_qc_masks_strict(mdata.mod['prot'])
    >>> mdata.mod['prot'].var['qc_pass'].sum(), len(passed)
    """
    X = adata_prot.layers[layer] if layer is not None else adata_prot.X
    n_cells = adata_prot.n_obs
    var_names = adata_prot.var_names.to_list()

    # allocate metrics
    frac_zero   = np.empty(adata_prot.n_vars, dtype=float)
    variance    = np.empty(adata_prot.n_vars, dtype=float)
    nz_frac     = np.empty(adata_prot.n_vars, dtype=float)
    n_unique    = np.empty(adata_prot.n_vars, dtype=float)
    r_robust    = np.empty(adata_prot.n_vars, dtype=float)  # p99 - p1
    med_vals    = np.empty(adata_prot.n_vars, dtype=float)
    reason      = np.empty(adata_prot.n_vars, dtype=object)

    def _dense_stats(col):
        """
        Compute stats for a dense column.

        Returns
        -------
        frac0, var, nz_frac_local, uniq, robust_range, median
        """
        if treat_nan_as_zero:
            col = np.where(np.isfinite(col), col, 0.0)
            finite = np.ones_like(col, dtype=bool)
        else:
            finite = np.isfinite(col)
            col = col[finite]
        n_eff = col.size
        if n_eff == 0:
            return 1.0, 0.0, 0.0, 0, 0.0, 0.0
        nz = np.count_nonzero(col)
        frac0 = (n_eff - nz) / n_eff
        s1 = col.sum(dtype=np.float64); s2 = (col.astype(np.float64)**2).sum(dtype=np.float64)
        mean = s1 / n_eff
        var = max(0.0, float(s2 / n_eff - mean*mean))
        if nz > 0:
            p1, p99 = np.percentile(col, [1, 99])
            med = np.median(col)
            uniq = int(np.unique(col[np.isfinite(col)]).size)
        else:
            p1 = p99 = med = 0.0
            uniq = int(np.unique(col[np.isfinite(col)]).size)
        nz_frac_local = nz / n_eff
        r = float(max(0.0, p99 - p1))
        return frac0, var, nz_frac_local, uniq, r, float(med)

    def _sparse_stats_csc(csc, j):
        """
        Compute stats for a CSC sparse column j.

        Returns
        -------
        frac0, var, nz_frac_local, uniq, robust_range, median
        """
        start, end = csc.indptr[j], csc.indptr[j+1]
        idx = csc.indices[start:end]
        dat = csc.data[start:end]

        finite_mask = np.isfinite(dat)
        idx = idx[finite_mask]
        dat = dat[finite_mask]

        if treat_nan_as_zero:
            n_eff = n_cells
        else:
            n_nan = ((~finite_mask).sum())
            n_eff = n_cells - int(n_nan)
        if n_eff <= 0:
            return 1.0, 0.0, 0.0, 0, 0.0, 0.0

        nz = len(dat)
        frac0 = (n_eff - nz) / n_eff

        s1 = dat.sum(dtype=np.float64)
        s2 = (dat.astype(np.float64)**2).sum(dtype=np.float64)
        mean = s1 / n_eff
        var = max(0.0, float(s2 / n_eff - mean*mean))
        nz_frac_local = nz / n_eff

        if nz > 0:
            p1 = np.percentile(dat, 1)
            p99 = np.percentile(dat, 99)
            med = float(np.median(dat))
            uniq = int(np.unique(dat).size)
            r = float(max(0.0, p99 - p1))
        else:
            p1 = p99 = med = 0.0
            uniq = 0
            r = 0.0
        return frac0, var, nz_frac_local, uniq, r, med

    if sparse.issparse(X):
        X = X.tocsc(copy=False)
        for j in range(X.shape[1]):
            fz, vr, nzf, uniq, r, med = _sparse_stats_csc(X, j)
            frac_zero[j], variance[j], nz_frac[j], n_unique[j], r_robust[j], med_vals[j] = fz, vr, nzf, uniq, r, med
    else:
        for j in range(X.shape[1]):
            col = np.asarray(X[:, j]).ravel()
            fz, vr, nzf, uniq, r, med = _dense_stats(col)
            frac_zero[j], variance[j], nz_frac[j], n_unique[j], r_robust[j], med_vals[j] = fz, vr, nzf, uniq, r, med

    # rules
    rule_A = variance <= var_eps
    rule_B = frac_zero >= zero_thresh
    rule_C = nz_frac < min_nonzero_frac
    rule_D = n_unique <= 1
    rule_E = r_robust < abs_range_eps
    rule_F = r_robust < (rel_range_eps * (med_vals + 1.0))

    failed_mask = rule_A | rule_B | rule_C | rule_D | rule_E | rule_F
    passed_mask = ~failed_mask

    reason[:] = ""
    for j in range(len(var_names)):
        rs = []
        if rule_A[j]: rs.append("singular")
        if rule_B[j]: rs.append("zeros≥thr")
        if rule_C[j]: rs.append("nonzero<min")
        if rule_D[j]: rs.append("≤1 unique")
        if rule_E[j]: rs.append("tiny_abs_range")
        if rule_F[j]: rs.append("tiny_rel_range")
        reason[j] = "pass" if not rs else "+".join(rs)

    if store:
        adata_prot.var["qc_frac_zero"]   = frac_zero
        adata_prot.var["qc_nonzero_frac"]= nz_frac
        adata_prot.var["qc_variance"]    = variance
        adata_prot.var["qc_n_unique"]    = n_unique
        adata_prot.var["qc_rrobust"]     = r_robust
        adata_prot.var["qc_median"]      = med_vals
        adata_prot.var["qc_fail"]        = failed_mask
        adata_prot.var["qc_pass"]        = passed_mask
        adata_prot.var["qc_fail_reason"] = reason

    passed_markers = [var_names[i] for i in np.where(passed_mask)[0]]
    failed_markers = [var_names[i] for i in np.where(failed_mask)[0]]
    return passed_mask, failed_mask, passed_markers, failed_markers


def show_marker_qc(marker):
    """
    Print QC metrics for a single protein marker.

    Notes
    -----
    This function expects a global `prot` AnnData object in scope:
    `prot = mdata.mod['prot']`. It simply prints a few `.var[...]` columns.

    Parameters
    ----------
    marker : str
        Marker/feature name present in `prot.var_names`.

    Returns
    -------
    None

    Examples
    --------
    >>> # Assuming `prot = mdata.mod['prot']` exists in your session
    >>> show_marker_qc("prot_CXCR3")
    """
    cols = ["qc_frac_zero","qc_nonzero_frac","qc_variance",
            "qc_n_unique","qc_rrobust","qc_median","qc_fail_reason","qc_pass"]
    print(marker, "→")
    print(prot.var.loc[marker, cols])  # noqa: F821 (prot is expected to exist in user scope)


# =============================================================================
# [DOUBLETS]
# =============================================================================
def run_scrublet_scores(i, rna, force=False):
    """
    Run Scrublet doublet detection on an RNA AnnData (if not already present).

    Parameters
    ----------
    i : Any
        Label used for printing/logging which sample or index is running.
    rna : AnnData
        RNA AnnData object. Will write results into `.obs` and `.uns['scrublet']`.
    force : bool, default: False
        If True, re-run Scrublet even if doublet info is already present.

    Returns
    -------
    tuple
        (i, obs_df_or_None, scrublet_uns_or_None)
        Where `obs_df` includes columns ["doublet_score", "predicted_doublet"].

    Notes
    -----
    Uses `sc.pp.scrublet(...)`. Leaves data in-place (`copy=False`).

    Examples
    --------
    >>> idx, df, meta = run_scrublet_scores("S1", rna, force=False)
    >>> if df is not None:
    ...     rna.obs[["doublet_score","predicted_doublet"]].head()
    """
    print(f"Sample {i}: {rna.n_obs} cells")

    if rna.X is None or rna.n_obs == 0:
        print(f"Sample {i}: Invalid or empty AnnData")
        return i, None, None

    scrublet_ran = (
        "doublet_score" in rna.obs and
        "predicted_doublet" in rna.obs and
        "scrublet" in rna.uns and
        "doublet_scores_sim" in rna.uns["scrublet"]
    )

    if scrublet_ran and not force:
        print(f"Sample {i}: Scrublet already run, skipping.")
        return i, rna.obs[["doublet_score", "predicted_doublet"]].copy(), rna.uns["scrublet"]

    try:
        sc.pp.scrublet(
            rna,
            sim_doublet_ratio=2,
            expected_doublet_rate=0.1,
            stdev_doublet_rate=0.05,
            normalize_variance=True,
            log_transform=False,
            n_prin_comps=30,
            verbose=True,
            copy=False,
            random_state=42
        )
        print(f"Sample {i}: Scrublet run complete.")
        return i, rna.obs[["doublet_score", "predicted_doublet"]].copy(), rna.uns["scrublet"]
    except Exception as e:
        print(f"Sample {i}: Error running Scrublet → {e}")
        return i, None, None


# =============================================================================
# [CLEANUP / STRIPPING]
# =============================================================================
def remove_elements(data, obs_list=None, uns_list=None, obsm_list=None, varm_list=None, obsp_list=None):
    """
    Remove matching keys from a COPY of an AnnData object's containers.

    Parameters
    ----------
    data : AnnData
        The AnnData object from which elements will be removed (on a copy).
    obs_list : list of str, optional
        Regex patterns to match and remove keys from `data.obs`.
    uns_list : list of str, optional
        Regex patterns to match and remove keys from `data.uns`.
    obsm_list : list of str, optional
        Regex patterns to match and remove keys from `data.obsm`.
    varm_list : list of str, optional
        Regex patterns to match and remove keys from `data.varm`.
    obsp_list : list of str, optional
        Regex patterns to match and remove keys from `data.obsp`.

    Returns
    -------
    AnnData
        A *copy* of the original AnnData object with specified elements removed.

    Examples
    --------
    >>> slim = remove_elements(
    ...     data,
    ...     obs_list=['^leiden', 'cluster'],
    ...     uns_list=['scvi', 'neighbors'],
    ...     obsm_list=['^X_umap']
    ... )
    """
    data_copy = data.copy()

    # Remove from obs
    if obs_list:
        for obs_str in obs_list:
            re_obs = re.compile(obs_str)
            obs_to_remove = [obs for obs in data_copy.obs_keys() if re.search(re_obs, obs)]
            for obs in obs_to_remove:
                del data_copy.obs[obs]

    # Remove from uns
    if uns_list:
        for uns_str in uns_list:
            re_uns = re.compile(uns_str)
            uns_to_remove = [uns for uns in data_copy.uns_keys() if re.search(re_uns, uns)]
            for uns in uns_to_remove:
                del data_copy.uns[uns]

    # Remove from obsm
    if obsm_list:
        for obsm_str in obsm_list:
            re_obsm = re.compile(obsm_str)
            obsm_to_remove = [obsm for obsm in data_copy.obsm_keys() if re.search(re_obsm, obsm)]
            for obsm in obsm_to_remove:
                del data_copy.obsm[obsm]

    # Remove from varm
    if varm_list:
        for varm_str in varm_list:
            re_varm = re.compile(varm_str)
            varm_to_remove = [varm for varm in data_copy.varm_keys() if re.search(re_varm, varm)]
            for varm in varm_to_remove:
                del data_copy.varm[varm]

    # Remove from obsp
    if obsp_list:
        for obsp_str in obsp_list:
            re_obsp = re.compile(obsp_str)
            obsp_to_remove = [obsp for obsp in list(data_copy.obsp) if re.search(re_obsp, obsp)]
            for obsp in obsp_to_remove:
                del data_copy.obsp[obsp]

    return data_copy


# =============================================================================
# [DIMRED / CLUSTERING / TOPOLOGY]
# =============================================================================
def umap_optimizer(adata, min_dist_spread_tuple: tuple, neighbors, level: str, init_pos: str):
    """
    Generate multiple UMAP embeddings across a grid of (min_dist, spread).

    Results are written to `.obsm` with keys:
    `X_umap_{min_dist}_{spread}_{level}`

    Parameters
    ----------
    adata : AnnData
        Target object with `.uns[neighbors]` computed and `.obsm['X_umap']` writable.
    min_dist_spread_tuple : tuple of (float, float)
        Sequence of (min_dist, spread) values to try.
    neighbors : str
        Key to the neighbors graph in `adata.uns` (`neighbors_key`).
    level : str
        Tag appended to the `.obsm` key.
    init_pos : {"spectral", "paga", ...}
        UMAP initialization strategy (passed to `sc.tl.umap`).

    Returns
    -------
    None
        Embeddings stored in `adata.obsm[...]`.

    Examples
    --------
    >>> grid = [(0.1, 1.0), (0.3, 1.2), (0.5, 1.0)]
    >>> umap_optimizer(adata, grid, neighbors="neighbors", level="rna", init_pos="spectral")
    """
    for d, s in min_dist_spread_tuple:
        min_dist = str(d)
        spread = str(s)
        embedding_label = 'X_umap_' + min_dist + '_' + spread + '_' + level
        sc.tl.umap(
            adata,
            min_dist=d,
            spread=s,
            n_components=3,
            maxiter=None,
            alpha=1.0,
            gamma=1.0,
            negative_sample_rate=5,
            init_pos=init_pos,
            random_state=42,
            copy=False,
            method='umap',
            neighbors_key=neighbors
        )
        adata.obsm[embedding_label] = adata.obsm['X_umap'].copy()


def leiden_optimizer(adata, umap_layer, resolutions: list, level: str, neighbors):
    """
    Compute Leiden clusters over a set of resolutions and plot on a given UMAP.

    Parameters
    ----------
    adata : AnnData
        Object with neighbors computed and UMAP embeddings in `.obsm`.
    umap_layer : str
        `.obsm` key used as plotting basis (e.g., 'X_umap_0.5_1.0_lvl_rna').
    resolutions : list[float]
        Resolutions to evaluate for Leiden clustering.
    level : str
        Tag used in the obs key name: 'leiden_{res}_lvl_{level}'.
    neighbors : str
        Neighbors key to use for clustering.

    Returns
    -------
    None
        Writes categorical clustering labels to `.obs[...]` and shows plots.

    Examples
    --------
    >>> leiden_optimizer(adata, "X_umap_0.5_1.0_lvl_rna", [0.5, 1.0, 1.5], "rna", neighbors="neighbors")
    """
    for i in resolutions:
        reso_label = str(i)
        key_label = 'leiden_' + reso_label + '_lvl_' + level
        sc.tl.leiden(
            adata,
            resolution=i,
            neighbors_key=neighbors,
            key_added=key_label,
            random_state=42,
            copy=False
        )
        sc.pl.embedding(
            adata,
            basis=umap_layer,
            color=key_label,
            legend_loc="on data",
            size=30
        )


def generate_processed_PAGA_UMAP(adata: ad.AnnData.__class__,
                                 leiden_to_use: str,
                                 neighbor_key: str,
                                 min_dist_spread_values: list[list[str]],
                                 level: str):
    """
    Compute PAGA on a given clustering and seed UMAPs using PAGA init across a grid.

    Parameters
    ----------
    adata : AnnData
        Object with neighbors/clusters and UMAP ready to be computed.
    leiden_to_use : str
        Column in `.obs` with cluster labels to group PAGA (e.g., 'leiden_1.0').
    neighbor_key : str
        Key to neighbors in `.uns`.
    min_dist_spread_values : list[list[str]]
        Sequence of [min_dist, spread] (strings accepted; converted to float).
    level : str
        Tag appended to UMAP `.obsm` keys.

    Returns
    -------
    None
        Shows a PAGA plot and writes multiple UMAP embeddings in `.obsm`.

    Examples
    --------
    >>> generate_processed_PAGA_UMAP(
    ...     adata, "leiden_1.0", "neighbors",
    ...     min_dist_spread_values=[["0.3","1.0"],["0.5","1.2"]],
    ...     level="rna"
    ... )
    """
    sc.tl.paga(adata, neighbors_key=neighbor_key, groups=leiden_to_use)
    sc.pl.paga(adata)
    min_dist_spread_values = [[float(value) for value in sublist] for sublist in min_dist_spread_values]
    umap_optimizer(
        adata,
        init_pos='paga',
        min_dist_spread_tuple=min_dist_spread_values,
        neighbors=neighbor_key,
        level=level
    )


def run_umap_leiden_grid(
    mdata,
    *,
    neighbors_key: str = "combined_wnn",
    key_prefix: str = "hm",
    name_tag: Optional[str] = None,
    include_k_in_tag: bool = True,
    k_label: str = "nn",
    modality: Optional[str] = "auto",
    store_in: str = "auto",  # {'auto','parent','modality','both','same_as_target'}
    run_umap: bool = True,
    run_leiden: bool = True,
    n_components: int = 3,
    umap_params: Optional[Union[Tuple[float,float], Sequence[Tuple[float,float]]]] = (0.5, 1.0),
    init_pos: Union[str, np.ndarray] = "spectral",
    random_state: int = 42,
    resolutions: Union[float, Sequence[float]] = 1.0,
    prefer_rapids: Optional[bool] = None,
    verbose: bool = True,
):
    """
    Grid-search UMAP and/or Leiden with optional RAPIDS acceleration, preserving results.

    Writes UMAPs to `.obsm["X_umap_{tag}_{min}_{spread}"]` and Leiden to
    `.obs["leiden_{tag}_{res}"]` on the chosen destination(s).

    Parameters
    ----------
    mdata : MuData or AnnData
        Target data (MuData recommended). If AnnData, computations occur on it directly.
    neighbors_key : str, default: "combined_wnn"
        Key under `.uns` where neighbors info resides.
    key_prefix : str, default: "hm"
        Prefix for the generated tag; final tag may include inferred neighbor `k`.
    name_tag : str or None, default: None
        If provided, overrides automatic tag generation.
    include_k_in_tag : bool, default: True
        If True and neighbor `k` can be inferred, include it in the tag.
    k_label : str, default: "nn"
        Label used when adding neighbor `k` into tag, e.g., "nn15".
    modality : {"auto", None, <mod_name>}, default: "auto"
        - "auto"/None : compute on parent object (MuData itself)
        - "<mod_name>": compute on mdata.mod[mod_name] but optionally store elsewhere
    store_in : {"auto","parent","modality","both","same_as_target"}, default: "auto"
        Where to write the resulting keys (parent MuData, the modality, or both).
    run_umap : bool, default: True
        Whether to compute UMAPs across `umap_params`.
    run_leiden : bool, default: True
        Whether to compute Leiden across `resolutions`.
    n_components : int, default: 3
        UMAP number of components.
    umap_params : (float, float) or sequence of (float, float), default: (0.5, 1.0)
        Grid of (min_dist, spread) for UMAP.
    init_pos : str or np.ndarray, default: "spectral"
        UMAP initialization.
    random_state : int, default: 42
        Random seed.
    resolutions : float or sequence of float, default: 1.0
        Leiden resolutions to evaluate.
    prefer_rapids : bool or None, default: None
        If True, require RAPIDS; if False, force Scanpy; if None, prefer RAPIDS if available.
    verbose : bool, default: True
        Print backend/step information.

    Returns
    -------
    dict
        Keys:
        - "umap_obsm_keys": list[str]
        - "leiden_obs_keys": list[str]
        - "umap_plot_basis": list[str]
        - "leiden_plot_keys": list[str]
        - "tag": str
        - "n_neighbors": int or None
        - "stored_in": list[tuple[str, Optional[str]]]
        - "backend_umap": {"scanpy","rapids"}
        - "backend_leiden": {"scanpy","rapids"}

    Examples
    --------
    >>> out = run_umap_leiden_grid(
    ...     mdata,
    ...     neighbors_key="combined_wnn",
    ...     modality="auto",
    ...     umap_params=[(0.3, 1.0), (0.5, 1.2)],
    ...     resolutions=[0.5, 1.0, 1.5],
    ...     prefer_rapids=None,  # use RAPIDS if available, else scanpy
    ... )
    >>> out["umap_obsm_keys"], out["leiden_obs_keys"]
    """
    is_mudata = hasattr(mdata, "mod")
    if is_mudata:
        if modality in ("auto", "parent", None):
            compute_target, compute_mod = mdata, None
        else:
            if modality not in mdata.mod:
                raise KeyError(f"Modality '{modality}' not found")
            compute_target, compute_mod = mdata.mod[modality], modality
    else:
        compute_target, compute_mod = mdata, None

    if neighbors_key not in compute_target.uns:
        raise KeyError(f"neighbors_key '{neighbors_key}' not found")

    nn = None
    try:
        params = compute_target.uns[neighbors_key].get("params", {})
        nn = params.get("n_neighbors", None)
        if nn is None:
            for k in ("n_neighbors", "n_neighbors_", "neighbors"):
                v = compute_target.uns[neighbors_key].get(k, None)
                if isinstance(v, (int, np.integer)):
                    nn = int(v); break
    except Exception:
        pass

    if name_tag is not None:
        tag = str(name_tag)
    else:
        parts = [str(key_prefix)] if key_prefix else []
        if include_k_in_tag and nn is not None: parts.append(f"{k_label}{int(nn)}")
        tag = "_".join(parts) if parts else "grid"

    def _auto_dest():
        if is_mudata and compute_mod is not None: return [("modality", compute_mod)]
        return [("parent", None)]
    if store_in == "auto": dests = _auto_dest()
    elif store_in == "parent": dests = [("parent", None)]
    elif store_in == "modality":
        if not (is_mudata and modality not in ("auto", "parent", None)):
            raise ValueError("store_in='modality' requires a modality name.")
        dests = [("modality", modality)]
    elif store_in == "both":
        if is_mudata and modality not in ("auto", "parent", None):
            dests = [("parent", None), ("modality", modality)]
        else: dests = [("parent", None)]
    elif store_in == "same_as_target": dests = [("modality", compute_mod)] if compute_mod else [("parent", None)]
    else: raise ValueError("Invalid store_in")

    def _numfmt(x: float) -> str: return str(x).replace(".", "_")
    def umap_name(min_dist, spread) -> str: return f"X_umap_{tag}_{_numfmt(min_dist)}_{_numfmt(spread)}"
    def leiden_name(res) -> str: return f"leiden_{tag}_{_numfmt(res)}"

    def _write_umap(embed, name):
        for kind, mm in dests:
            if not is_mudata: compute_target.obsm[name] = embed
            elif kind == "parent": mdata.obsm[name] = embed
            else: mdata.mod[mm].obsm[name] = embed

    def _write_leiden(series, name):
        cat = series.astype("category")
        for kind, mm in dests:
            if not is_mudata: compute_target.obs[name] = cat
            elif kind == "parent": mdata.obs[name] = cat
            else: mdata.mod[mm].obs[name] = cat

    # backend selection (prefer RAPIDS)
    import importlib
    tl_umap = tl_leiden = None
    backend_umap = backend_leiden = "scanpy"
    rapids_ok = prefer_rapids is not False
    try:
        if rapids_ok:
            rsc = importlib.import_module("rapids_singlecell")
            if hasattr(rsc.tl, "umap"): tl_umap, backend_umap = rsc.tl.umap, "rapids"
            if hasattr(rsc.tl, "leiden"): tl_leiden, backend_leiden = rsc.tl.leiden, "rapids"
    except Exception as e:
        if prefer_rapids is True:
            raise RuntimeError(f"RAPIDS unavailable: {e}")
    if tl_umap is None: tl_umap, backend_umap = sc.tl.umap, "scanpy"
    if tl_leiden is None: tl_leiden, backend_leiden = sc.tl.leiden, "scanpy"
    if verbose:
        print(f"[grid] Using UMAP backend: {backend_umap}")
        print(f"[grid] Using Leiden backend: {backend_leiden}")

    # normalize grids
    umap_grid = []
    if run_umap:
        umap_grid = [umap_params] if isinstance(umap_params, tuple) else list(umap_params or [])
    res_list = []
    if run_leiden:
        res_list = [float(resolutions)] if not isinstance(resolutions, (list, tuple, np.ndarray)) else [float(r) for r in resolutions]

    # helper: filter kwargs by signature
    def _call_with_filtered(func, **kwargs):
        sig = inspect.signature(func)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(compute_target, **allowed)

    made_umaps, made_leidens = [], []

    # UMAP
    for (min_dist, spread) in umap_grid:
        if verbose: print(f"[grid] UMAP: tag={tag}, min_dist={min_dist}, spread={spread}")
        umap_kwargs = dict(
            min_dist=float(min_dist),
            spread=float(spread),
            n_components=int(n_components),
            init_pos=init_pos,
            random_state=int(random_state),
            neighbors_key=neighbors_key,
            copy=False,
        )
        if backend_umap == "rapids":
            umap_kwargs["method"] = "umap"  # filtered out automatically if unsupported
        _call_with_filtered(tl_umap, **umap_kwargs)
        out_key = umap_name(min_dist, spread)
        _write_umap(compute_target.obsm["X_umap"].copy(), out_key)
        made_umaps.append(out_key)

    # Leiden
    for res in res_list:
        if verbose: print(f"[grid] Leiden: tag={tag}, resolution={res}")
        obs_key = leiden_name(res)
        _call_with_filtered(
            tl_leiden,
            resolution=float(res),
            key_added=obs_key,
            neighbors_key=neighbors_key,
            random_state=int(random_state),
            copy=False,
        )
        _write_leiden(compute_target.obs[obs_key], obs_key)
        made_leidens.append(obs_key)

    umap_plot_basis = [f"{compute_mod}:{k}" if compute_mod else k for k in made_umaps] if is_mudata else made_umaps
    leiden_plot_keys = [f"{compute_mod}:{k}" if compute_mod else k for k in made_leidens] if is_mudata else made_leidens
    return {
        "umap_obsm_keys": made_umaps,
        "leiden_obs_keys": made_leidens,
        "umap_plot_basis": umap_plot_basis,
        "leiden_plot_keys": leiden_plot_keys,
        "tag": tag,
        "n_neighbors": nn,
        "stored_in": dests,
        "backend_umap": backend_umap,
        "backend_leiden": backend_leiden,
    }


# =============================================================================
# [ADT / PROTEIN PROCESSING]
# =============================================================================
def clr_prot(mdata, obsm_key="X_clr"):
    """
    Apply centered log-ratio (CLR) normalization to the protein modality.

    The CLR is computed from `log1p(X)` per cell and mean-centered row-wise,
    stored in `mdata.mod['prot'].obsm[obsm_key]`.

    Parameters
    ----------
    mdata : MuData
        MuData object with a `'prot'` modality.
    obsm_key : str, default: "X_clr"
        `.obsm` key to store the CLR matrix.

    Returns
    -------
    None
        Writes to `mdata.mod['prot'].obsm[obsm_key]`.

    Examples
    --------
    >>> clr_prot(mdata, obsm_key="X_clr")
    >>> mdata.mod['prot'].obsm["X_clr"].shape
    """
    prot = mdata.mod['prot']

    # Get matrix and ensure it's dense
    X = prot.X.toarray() if hasattr(prot.X, 'toarray') else prot.X

    # Add pseudocount to avoid log(0)
    X = np.log1p(X)

    # CLR: subtract mean log1p value for each cell (row-wise)
    X_clr = X - X.mean(axis=1, keepdims=True)

    # Store in .obsm
    prot.obsm[obsm_key] = X_clr


def run_gmm_thresholding(
    mdata_or_list,
    markers=None,
    obsm_key="X_clr",
    per_sample=True,
    sample_col="sample_id",
    positive_only=True,           # mimic ThresholdR (>0) and speed up
    max_k=3,
    engine="hist",                # "hist" (fast, binned EM) or "sklearn" (exact)
    global_k="pooled_bic",        # decide k once per marker using pooled data
    n_bins=256,                   # for hist engine
    subsample=None,               # cap pooled cells, e.g. 50000
    max_iter=150,
    tol=1e-3,
    random_state=42,
    n_jobs=8,                     # parallelize per sample
    warm_start=True,              # sklearn: init from pooled fit
    save_plots_dir=None,          # ridgeline PDF per marker
    verbose=1,                    # 0=silent, 1=marker+dots, 2=debug
    # ---- guards & stability ----
    min_points=8,         # require at least this many points per (marker,sample)
    min_unique=2,         # at least this many unique values
    min_nonzero_bins=2,   # for histogram engine: non-empty bins needed
    reg_covar=1e-6,       # sklearn GMM regularization
    # ---- write-back options ----
    write_layer=True,
    prot_layer_name=None,         # default: f"denoised_from_{obsm_key}"
    write_modality=True,
    modality_name="denoised_prot",
):
    """
    Fast, marker-first GMM thresholding with pooled-BIC model order selection.

    Workflow
    --------
    1) For each marker, pool all cells (optionally subsample), choose best k (1..max_k)
       by BIC (histogram-EM or sklearn).
    2) Fit that k per-sample to get (weights, means, sigmas).
    3) Derive thresholds from either:
         - component intersection(s) (lowest cut)
         - or mean+3*SD of the first component
       pick the *more conservative* (final threshold is min intersection vs mean+3SD).
    4) Write back a thresholded matrix (values ≤ thr → 0; else original CLR)
       into:
         - `prot.layers[prot_layer_name]`
         - `mdata.mod[modality_name].X` (clone of protein modality with thresholded X)
    5) Optionally create ridgeline PDFs per marker.

    Parameters
    ----------
    mdata_or_list : MuData or list[MuData]
        - Single MuData with many samples in `mod['prot'].obs[sample_col]`, or
        - A list where each MuData is a separate sample.
    markers : list[str] or None
        Markers to process. If None, use all markers from the first object.
    obsm_key : str, default: "X_clr"
        Key to CLR matrix in `prot.obsm`.
    per_sample : bool, default: True
        If True, split fits by `sample_col` in `.obs`; else treat whole object as one sample.
    sample_col : str, default: "sample_id"
        Column in `prot.obs` defining sample group membership.
    positive_only : bool, default: True
        Use only values > 0 for fitting.
    max_k : int, default: 3
        Max Gaussian components in GMM.
    engine : {"hist","sklearn"}, default: "hist"
        - "hist": histogram-EM (fast, robust), using `n_bins`.
        - "sklearn": exact GaussianMixture fits.
    global_k : {"pooled_bic"}, default: "pooled_bic"
        Determine a single k per-marker based on pooled data; that k is used per sample.
    n_bins : int, default: 256
        Number of bins for histogram engine.
    subsample : int or None
        If set, subsample pooled cells per marker (e.g., 50000).
    max_iter : int, default: 150
        EM maximum iterations.
    tol : float, default: 1e-3
        EM convergence tolerance.
    random_state : int, default: 42
        Seed for reproducibility.
    n_jobs : int, default: 8
        Parallel jobs for per-sample fits (hist or sklearn).
    warm_start : bool, default: True
        For sklearn engine, initialize from pooled fit when shapes match.
    save_plots_dir : str or None
        Directory to save ridgeline PDFs per marker.
    verbose : {0,1,2}, default: 1
        Print progress. Each '.' is a successful multi-component fit; 'x' indicates skip/basic fit.

    Guards & Stability
    ------------------
    min_points : int, default: 8
        Minimum number of points per-sample; otherwise skip (k=1 fallback).
    min_unique : int, default: 2
        Require at least 2 unique values per-sample.
    min_nonzero_bins : int, default: 2
        Histogram engine requires ≥ this many non-empty bins.
    reg_covar : float, default: 1e-6
        Sklearn GMM covariance regularization.

    Write-back
    ----------
    write_layer : bool, default: True
        Write thresholded matrix to `prot.layers[prot_layer_name]`.
    prot_layer_name : str or None
        If None, uses f"denoised_from_{obsm_key}".
    write_modality : bool, default: True
        Create `mdata.mod[modality_name]` with `.X` set to thresholded matrix.
    modality_name : str, default: "denoised_prot"
        Name of the cloned protein modality.

    Returns
    -------
    results : dict[str, pandas.DataFrame]
        marker -> DataFrame(index=sample, columns:
          ["best_k","weights","mus","sigmas","threshold","threshold_type",
           "cut","thr_mean3sd"])
    summary_df : pandas.DataFrame
        Per-(marker, sample) status table with skip reasons.

    Examples
    --------
    >>> # Ensure CLR exists first:
    >>> clr_prot(mdata, obsm_key="X_clr")
    >>> results, summary = run_gmm_thresholding(
    ...     mdata, markers=["CD3", "CD19"], obsm_key="X_clr",
    ...     per_sample=True, sample_col="sample_id",
    ...     engine="hist", max_k=3, save_plots_dir="./ridgelines"
    ... )
    >>> summary.head()
    """
    # ------------------ helpers ------------------
    def _normal_pdf(x, mu, sigma):
        """Univariate normal PDF."""
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2*np.pi) * sigma)

    def _em_histogram(xc, cc, k, max_iter=150, tol=1e-3, seed=42):
        """
        Weighted EM on histogram centers 'xc' with counts 'cc'.

        Parameters
        ----------
        xc : np.ndarray
            Bin centers (shape: [B]).
        cc : np.ndarray
            Counts per bin (shape: [B]).
        k : int
            Number of Gaussian components.

        Returns
        -------
        weights, mus, sigmas, loglik : tuple
            Arrays of length k and the final log-likelihood (float).

        Notes
        -----
        Initializes means via weighted quantiles, variances by range/6k.
        """
        qs = np.linspace(0.1, 0.9, k)
        cdf = np.cumsum(cc) / (cc.sum() + 1e-12)
        mus = np.interp(qs, cdf, xc)
        sigmas = np.full(k, max(1e-3, (xc.max() - xc.min()) / (6*k)))
        weights = np.full(k, 1.0/k)
        prev_ll = -np.inf
        for _ in range(max_iter):
            comp = np.stack([weights[j] * _normal_pdf(xc, mus[j], sigmas[j]) for j in range(k)], axis=1)
            denom = comp.sum(axis=1, keepdims=True) + 1e-12
            resp = comp / denom
            Nk = (cc[:, None] * resp).sum(axis=0) + 1e-12
            weights = Nk / Nk.sum()
            mus = (cc[:, None] * resp * xc[:, None]).sum(axis=0) / Nk
            var = (cc[:, None] * resp * (xc[:, None] - mus)**2).sum(axis=0) / Nk
            sigmas = np.sqrt(np.clip(var, 1e-6, None))
            ll = np.sum(cc * np.log(np.maximum(denom.squeeze(), 1e-300)))
            if ll - prev_ll < tol * (abs(prev_ll) if prev_ll != 0 else 1):
                break
            prev_ll = ll
        return weights, mus, sigmas, float(ll)

    def _bic(ll, k, n_eff):
        """Bayesian Information Criterion for 1D k-GMM."""
        p = 3*k - 1
        return -2*ll + p * np.log(max(n_eff, 1))

    def _find_intersections(mu1, s1, w1, mu2, s2, w2):
        """
        Find intersections of two 1D Gaussian components.

        Returns
        -------
        list[float]
            Real intersection points (could be 0, 1, or 2).
        """
        a = 1/(2*s1**2) - 1/(2*s2**2)
        b = mu2/(s2**2) - mu1/(s1**2)
        c = (mu1**2)/(2*s1**2) - (mu2**2)/(2*s2**2) - np.log((s2*w1)/(s1*w2))
        disc = b*b - 4*a*c
        if disc < 0:
            return []
        if np.isclose(a, 0):
            return [-c/b]
        r1 = (-b + np.sqrt(disc)) / (2*a)
        r2 = (-b - np.sqrt(disc)) / (2*a)
        return [r1, r2]

    def _compute_thresholds(mus, sigmas, weights, lo, hi):
        """
        Choose final threshold = min(intersection, mean+3SD of first component)."""
        order = np.argsort(mus)
        mus, sigmas, weights = np.array(mus)[order], np.array(sigmas)[order], np.array(weights)[order]
        thr_mean3sd = mus[0] + 3*sigmas[0]
        inter = []
        for i in range(len(mus)-1):
            inter += _find_intersections(mus[i], sigmas[i], weights[i], mus[i+1], sigmas[i+1], weights[i+1])
        inter = [x for x in inter if np.isfinite(x) and lo <= x <= hi]
        cut = min(inter) if inter else np.nan
        if np.isfinite(cut) and cut < thr_mean3sd:
            return cut, thr_mean3sd, cut, "intersection"
        else:
            return cut, thr_mean3sd, thr_mean3sd, "mean+3SD"

    def _collect_samples(mdl, marker, obsm_key, per_sample, sample_col, positive_only):
        """
        OrderedDict: sample_name -> 1D array of CLR values for `marker`.
        """
        out = OrderedDict()
        def _label(obj, idx):
            try:
                return str(obj.uns.get("name", f"obj{idx}"))
            except Exception:
                return f"obj{idx}"
        if isinstance(mdl, (list, tuple)):
            for i, md in enumerate(mdl, start=1):
                prot = md.mod["prot"]
                if marker not in prot.var_names:
                    continue
                j = prot.var_names.get_loc(marker)
                X = prot.obsm[obsm_key]
                vals = np.asarray(X[:, j], dtype=np.float32)
                if positive_only: vals = vals[vals > 0]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    out[_label(md, i)] = vals
        else:
            md = mdl
            prot = md.mod["prot"]
            if marker not in prot.var_names:
                return out
            j = prot.var_names.get_loc(marker)
            X = prot.obsm[obsm_key]
            if per_sample and sample_col in prot.obs.columns:
                groups = prot.obs[sample_col].astype(str).values
                for g in pd.unique(groups):
                    idx = np.where(groups == g)[0]
                    vals = np.asarray(X[idx, j], dtype=np.float32)
                    if positive_only: vals = vals[vals > 0]
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        out[g] = vals
            else:
                vals = np.asarray(X[:, j], dtype=np.float32)
                if positive_only: vals = vals[vals > 0]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    out[_label(md, 1)] = vals
        return out

    def _ridgeplot_marker(sample2vals, sample2thrrow, marker, save_pdf=None, bins=60, alpha_hist=0.5, height=0.9):
        """
        Ridgeline-like stacked histograms with threshold lines per sample.
        """
        if not sample2vals: return
        items = list(sample2vals.items())
        lo = min(v.min() for _, v in items); hi = max(v.max() for _, v in items)
        pad = 0.02 * (hi - lo + 1e-9); x_lo, x_hi = lo-pad, hi+pad
        fig, ax = plt.subplots(figsize=(9, max(3, 0.6*len(items))))
        drew_final = drew_cut = drew_m3 = False
        for r, (sname, vals) in enumerate(items):
            offset = r * height
            counts, edges = np.histogram(vals, bins=bins, range=(x_lo, x_hi), density=True)
            centers = (edges[:-1] + edges[1:]) / 2.0
            ax.fill_between(centers, offset, counts+offset, alpha=alpha_hist, step="mid")
            row = sample2thrrow.get(sname)
            if row is not None:
                ymax = offset + (counts.max() if counts.size else 0.1)
                thr = float(row.get("threshold", np.nan))
                if np.isfinite(thr):
                    ln = ax.vlines(thr, offset, ymax, linestyles="-.", linewidth=1.5)
                    if not drew_final: ln.set_label("Final threshold"); drew_final = True
                if "cut" in row.index:
                    cut = float(row.get("cut", np.nan))
                    if np.isfinite(cut):
                        ln = ax.vlines(cut, offset, ymax, linestyles="--", linewidth=1.0)
                        if not drew_cut: ln.set_label("Intersection"); drew_cut = True
                if "thr_mean3sd" in row.index:
                    m3 = float(row.get("thr_mean3sd", np.nan))
                    if np.isfinite(m3):
                        ln = ax.vlines(m3, offset, ymax, linestyles=":", linewidth=1.0)
                        if not drew_m3: ln.set_label("Mean+3SD"); drew_m3 = True
        yticks = [i*height + 0.5*height for i in range(len(items))]
        ax.set_yticks(yticks); ax.set_yticklabels([n for n,_ in items])
        ax.set_xlim(x_lo, x_hi); ax.set_xlabel(f"{marker} (CLR)"); ax.set_ylabel("samples")
        ax.set_title(f"Ridgeline — {marker}")
        h,l = ax.get_legend_handles_labels()
        if h: ax.legend(loc="upper right", frameon=False)
        plt.tight_layout()
        if save_pdf:
            os.makedirs(os.path.dirname(save_pdf) or ".", exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight"); plt.close(fig)
        else:
            plt.show()

    # ------------------ pick marker list ------------------
    if isinstance(mdata_or_list, (list, tuple)):
        base = mdata_or_list[0].mod["prot"]
        all_markers = base.var_names.tolist()
    else:
        all_markers = mdata_or_list.mod["prot"].var_names.tolist()
    markers = [m for m in (markers or all_markers) if m in all_markers]
    if not markers:
        raise ValueError("No valid markers to process.")
    if prot_layer_name is None:
        prot_layer_name = f"denoised_from_{obsm_key}"

    # ------------------ main loop ------------------
    results = {}
    summary_records = []  # collect (marker, sample, status, reason)

    # optional parallel
    if n_jobs != 1:
        from joblib import Parallel, delayed

    for mi, marker in enumerate(markers, start=1):
        sample2vals = _collect_samples(mdata_or_list, marker, obsm_key, per_sample, sample_col, positive_only)
        if not sample2vals:
            if verbose:
                print(f"[{mi}/{len(markers)}] {marker}: (no data)")
            continue
        pooled = np.concatenate(list(sample2vals.values()))
        if subsample and pooled.size > subsample:
            rng = np.random.default_rng(random_state)
            pooled = pooled[rng.choice(pooled.size, subsample, replace=False)]
        lo, hi = float(np.min(pooled)), float(np.max(pooled))

        # -------- global k and pooled params (for warm start) --------
        best_k = None
        pooled_params = None
        if global_k == "pooled_bic":
            if pooled.size < min_points or np.unique(pooled).size < min_unique:
                best_k = min(2, max_k)
                pooled_params = None
            else:
                if engine == "hist":
                    counts, edges = np.histogram(pooled, bins=n_bins, density=False)
                    centers = (edges[:-1] + edges[1:]) / 2.0
                    if (counts > 0).sum() < min_nonzero_bins:
                        best_k = min(2, max_k)
                        pooled_params = None
                    else:
                        cand = {}
                        for k in range(1, max_k+1):
                            w, mu, sg, ll = _em_histogram(centers, counts.astype(float), k, max_iter=max_iter, tol=tol, seed=random_state)
                            cand[k] = _bic(ll, k, counts.sum())
                        best_k = min(cand, key=cand.get)
                        # small 3rd component guard
                        w, mu, sg, _ = _em_histogram(centers, counts.astype(float), best_k, max_iter=max_iter, tol=tol, seed=random_state)
                        if best_k == 3 and w.min() < 0.05:
                            best_k = 2
                            w, mu, sg, _ = _em_histogram(centers, counts.astype(float), 2, max_iter=max_iter, tol=tol, seed=random_state)
                        pooled_params = (w, mu, sg)
                else:
                    Xp = pooled.reshape(-1,1)
                    cand = {}
                    fits = {}
                    for k in range(1, max_k+1):
                        g = GaussianMixture(n_components=k, random_state=random_state, max_iter=max_iter, tol=tol, n_init=1, reg_covar=reg_covar)
                        g.fit(Xp)
                        cand[k] = g.bic(Xp); fits[k] = g
                    best_k = min(cand, key=cand.get)
                    g = fits[best_k]
                    if best_k == 3 and g.weights_.min() < 0.05:
                        best_k = 2 if 2 in fits else best_k
                        g = fits.get(best_k, g)
                    pooled_params = (g.weights_.flatten(), g.means_.flatten(), np.sqrt(g.covariances_).flatten())
        else:
            best_k = max_k
            pooled_params = None
        if verbose:
            print(f"[{mi}/{len(markers)}] {marker} (k={best_k}): ", end="", flush=True)

        # -------- per-sample fitter --------
        sample_names = list(sample2vals.keys())
        vals_list = [sample2vals[s] for s in sample_names]

        def _fit_one(vals):
            """
            Fit single sample distribution for one marker and compute GMM params.

            Returns
            -------
            (k_i, w, mu, sg, status, reason)
            """
            # guards
            if vals.size < min_points:
                return (1, np.array([1.0]), np.array([np.nanmean(vals) if vals.size else 0.0]), np.array([np.nanstd(vals) if vals.size else 1e-3]),
                        "skip", "min_points")
            if np.unique(vals).size < min_unique:
                return (1, np.array([1.0]), np.array([float(np.mean(vals))]), np.array([float(np.std(vals) or 1e-3)]),
                        "skip", "min_unique")
            if engine == "hist":
                counts, edges = np.histogram(vals, bins=n_bins, density=False)
                centers = (edges[:-1] + edges[1:]) / 2.0
                if (counts > 0).sum() < min_nonzero_bins:
                    return (1, np.array([1.0]), np.array([float(np.mean(vals))]), np.array([float(np.std(vals) or 1e-3)]),
                            "skip", "nonzero_bins")
                k = best_k
                w, mu, sg, _ = _em_histogram(centers, counts.astype(float), k, max_iter=max_iter, tol=tol, seed=random_state)
                return (k, w, mu, sg, "fit", "")
            else:
                X = vals.reshape(-1,1)
                k = best_k
                try:
                    g = GaussianMixture(
                        n_components=k, random_state=random_state,
                        max_iter=max_iter, tol=tol, n_init=1, reg_covar=reg_covar
                    )
                    if warm_start and pooled_params is not None:
                        pw, pm, ps = pooled_params
                        if len(pw) == k:
                            g.weights_init = pw
                            g.means_init   = pm.reshape(-1,1)
                    g.fit(X)
                    return (k, g.weights_.flatten(), g.means_.flatten(), np.sqrt(g.covariances_).flatten(),
                            "fit", "")
                except Exception:
                    return (1, np.array([1.0]), np.array([float(np.mean(vals))]), np.array([float(np.std(vals) or 1e-3)]),
                            "skip", "gmm_error")
        # run fits (parallel/sequential) with progress dots/x
        out = []
        if n_jobs == 1:
            for _vals in vals_list:
                k_i, w, mu, sg, status, reason = _fit_one(_vals)
                out.append((k_i, w, mu, sg, status, reason))
                if verbose: print("." if status == "fit" and k_i > 1 else "x", end="", flush=True)
        else:
            from joblib import Parallel, delayed
            chunks = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_fit_one)(v) for v in vals_list)
            for (k_i, w, mu, sg, status, reason) in chunks:
                out.append((k_i, w, mu, sg, status, reason))
                if verbose: print("." if status == "fit" and k_i > 1 else "x", end="", flush=True)

        # -------- assemble rows + thresholds; log summary --------
        rows = []
        for sname, vals, (k_i, w, mu, sg, status, reason) in zip(sample_names, vals_list, out):
            if k_i == 1:
                cut = m3 = thr = np.nan; ttype = "none"
            else:
                cut, m3, thr, ttype = _compute_thresholds(mu, sg, w, lo, hi)
            rows.append({
                "sample": sname,
                "best_k": int(k_i),
                "weights": w, "mus": mu, "sigmas": sg,
                "threshold": thr, "threshold_type": ttype,
                "cut": cut, "thr_mean3sd": m3
            })
            summary_records.append({
                "marker": marker,
                "sample": sname,
                "status": status if k_i == 1 else "fit",
                "reason": reason if k_i == 1 else ""
            })
        df_m = pd.DataFrame(rows).set_index("sample")
        results[marker] = df_m

        if verbose:
            cnt_k = df_m["best_k"].value_counts().to_dict()
            cnt_t = df_m["threshold_type"].value_counts().to_dict()
            print(f"  done | k: {cnt_k} | type: {cnt_t}")
        if save_plots_dir:
            os.makedirs(save_plots_dir, exist_ok=True)
            sample2thrrow = {s: results[marker].loc[s] for s in results[marker].index}
            _ridgeplot_marker(sample2vals, sample2thrrow, marker,
                              save_pdf=os.path.join(save_plots_dir, f"{marker}_ridgeline.pdf"))

    # ------------------ summary table ------------------
    summary_df = pd.DataFrame(summary_records, columns=["marker", "sample", "status", "reason"])

    # ------------------ WRITE BACK: prot layer + new modality ------------------
    def _threshold_matrix_for_object(md, results, markers, obsm_key, per_sample, sample_col):
        """
        Build a thresholded (cells x markers) matrix for this object.
        """
        prot = md.mod["prot"]
        Xclr = prot.obsm[obsm_key]
        n_cells, n_markers_tot = Xclr.shape
        out = np.zeros_like(Xclr, dtype=Xclr.dtype)

        var_index = {m: i for i, m in enumerate(prot.var_names.tolist())}

        if per_sample and (sample_col in prot.obs.columns):
            groups = prot.obs[sample_col].astype(str).values
            uniq = pd.unique(groups)
            for m in markers:
                j = var_index.get(m, None)
                if j is None:
                    continue
                df_m = results.get(m, None)
                if df_m is None:
                    continue
                for g in uniq:
                    thr = df_m.loc[g, "threshold"] if (g in df_m.index) else np.nan
                    idx = (groups == g)
                    colvals = Xclr[idx, j]
                    if np.isfinite(thr):
                        out[idx, j] = np.where(colvals > thr, colvals, 0.0)
                    else:
                        out[idx, j] = 0.0
        else:
            sname = str(md.uns.get("name", "sample"))
            for m in markers:
                j = var_index.get(m, None)
                if j is None:
                    continue
                df_m = results.get(m, None)
                if df_m is None or (sname not in df_m.index):
                    thr = np.nan
                else:
                    thr = df_m.loc[sname, "threshold"]
                colvals = Xclr[:, j]
                if np.isfinite(thr):
                    out[:, j] = np.where(colvals > thr, colvals, 0.0)
                else:
                    out[:, j] = 0.0
        return out

    mlist = mdata_or_list if isinstance(mdata_or_list, (list, tuple)) else [mdata_or_list]
    for md in mlist:
        prot = md.mod["prot"]
        present_markers = [m for m in markers if m in prot.var_names]
        if not present_markers:
            continue
        thr_mat = _threshold_matrix_for_object(md, results, present_markers, obsm_key, per_sample, sample_col)

        if write_layer:
            prot.layers[prot_layer_name] = thr_mat

        if write_modality:
            deno = prot.copy()
            deno.X = thr_mat
            md.mod[modality_name] = deno

    if verbose:
        total = len(summary_df)
        ok = (summary_df["status"] == "fit").sum()
        skipped = total - ok
        print(f"\nSummary: {ok} fits, {skipped} skipped (of {total}).")
        if skipped:
            print(summary_df[summary_df["status"] != "fit"]["reason"].value_counts())

    return results, summary_df

# =============================================================================
# [VISUALIZATION]
# =============================================================================
def plot_umap_by_genes_and_subtypes(
    adata, genotypes_order, umap_basis, genes, b_cell_subtypes, figsize=(10, 5), point_size=30):
    """
    Grid of UMAPs: columns by genotype, rows by gene; limited to B cell subtypes.

    Parameters
    ----------
    adata : AnnData
        Contains UMAP in `.obsm[umap_basis]` and obs columns:
        - 'genotype' and 'B_cell_subtype'
    genotypes_order : list[str]
        Order of genotypes for columns.
    umap_basis : str
        `.obsm` key for plotting basis (e.g., "X_umap_0.5_0.8_2").
    genes : list[str]
        Genes to color by; one row per gene.
    b_cell_subtypes : list[str]
        Allowed values in `adata.obs['B_cell_subtype']`.
    figsize : tuple, default: (10,5)
        Figure size passed to `plt.subplots`.
    point_size : int, default: 30
        Scanpy scatter size.

    Returns
    -------
    None
        Shows Matplotlib figure(s).

    Examples
    --------
    >>> plot_umap_by_genes_and_subtypes(
    ...     adata, ["WT","KO"], "X_umap_0_5_1_0_lvl_rna", ["Iglv1","Iglc1"], ["B1","MZB"]
    ... )
    """
    import matplotlib.pyplot as plt

    with plt.rc_context({'figure.titlesize': 'large'}):
        genotypes = adata.obs['genotype'].unique()
        genotypes = pd.Categorical(genotypes, categories=genotypes_order, ordered=True)
        genotypes = genotypes.categories

        fig, axes = plt.subplots(
            len(genes), len(genotypes), figsize=figsize, sharex=True, sharey=True
        )
        for row, gene in enumerate(genes):
            for col, genotype in enumerate(genotypes):
                adata_filtered = adata[
                    (adata.obs['genotype'] == genotype) & (adata.obs["B_cell_subtype"].isin(b_cell_subtypes))
                ]
                sc.pl.embedding(
                    adata_filtered,
                    basis=umap_basis,
                    color=[gene],
                    legend_loc=None,
                    color_map="viridis",
                    size=point_size,
                    show=False,
                    ax=axes[row, col] if len(genes) > 1 else axes[col],
                )
                if row == 0:
                    axes[row, col].set_title(f"{genotype}")
                else:
                    axes[row, col].set_title("")
        for row, gene in enumerate(genes):
            fig.text(
                0.08,
                (len(genes) - row - 0.5) / len(genes),
                gene,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                rotation=0,
            )

        fig.tight_layout(rect=[0.1, 0.1, 0.95, 0.95])
        plt.show()

def histogram_kde_facet_vertical_multi(
    adata_or_mdata,
    markers,
    modality: str = "prot",
    layer: str | None = None,
    sample_col: str = "sample_id",
    bins: int = 60,
    # KDE behavior
    kde: bool = True,
    drop_zeros_for_kde: bool = True,
    kde_bw_adjust: float = 1.0,
    # Y-axis control to handle giant zero spikes
    stat: str = "density",
    y_cap_quantile: float = 0.98,
    # layout
    height: float = 1.2,
    aspect: float = 3.5,
    sharex: bool = False,
    sharey: bool = False,
    # thresholds
    add_thresholds: bool = True,
    title: str | None = None,
):
    """
    Vertical-facet histogram + KDE, rows=samples, cols=markers.

    - Works with MuData or AnnData
    - Uses `layer` if provided; otherwise `.X`
    - Per-facet y-axis capping to reveal peaks despite zero spikes
    - Optionally draws per-sample thresholds if available at:
        `adata.uns['adt_thresholds']` (DataFrame: rows=samples, cols=markers)

    Parameters
    ----------
    adata_or_mdata : AnnData or MuData
        Source container. If MuData, `modality` is used (default 'prot').
    markers : list[str]
        Markers to include.
    modality : str, default: "prot"
        If MuData provided, which modality to use.
    layer : str or None, default: None
        If not None and present in `.layers`, use it; else use `.X`.
    sample_col : str, default: "sample_id"
        Obs column giving sample labels (facets down rows).
    bins : int, default: 60
        Histogram bins.
    kde : bool, default: True
        Overlay KDE per facet.
    drop_zeros_for_kde : bool, default: True
        Remove zeros for KDE computation only.
    kde_bw_adjust : float, default: 1.0
        KDE bandwidth adjustment.
    stat : {"density","count"}, default: "density"
        Histogram stat to use.
    y_cap_quantile : float, default: 0.98
        Cap per-facet y-limits to this quantile of heights.
    height : float, default: 1.2
        Per-row height.
    aspect : float, default: 3.5
        Width/height ratio.
    sharex, sharey : bool, default: False
        FacetGrid axis sharing options.
    add_thresholds : bool, default: True
        Draw vertical threshold lines if table exists at `uns['adt_thresholds']`.
    title : str or None, default: None
        Optional suptitle.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        The constructed FacetGrid.

    Examples
    --------
    >>> g = histogram_kde_facet_vertical_multi(mdata, ["CD3","CD19"], modality="prot",
    ...                                        layer="denoised_from_X_clr", sample_col="sample_id")
    >>> g.fig.savefig("hist_kde_grid.pdf")
    """
    if hasattr(adata_or_mdata, "mod"):   # MuData
        adata = adata_or_mdata.mod[modality]
    else:
        adata = adata_or_mdata
    if sample_col not in adata.obs.columns:
        raise KeyError(f"Column '{sample_col}' not found in .obs")
    vals_list = []
    for m in markers:
        if m not in adata.var_names:
            continue
        if layer is not None and layer in adata.layers:
            arr = adata.layers[layer][:, adata.var_names.get_loc(m)]
        else:
            arr = adata[:, m].X
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        v = np.ravel(arr).astype(float)
        vals_list.append(pd.DataFrame({
            "value": v,
            "marker": m,
            sample_col: adata.obs[sample_col].astype(str).values
        }))
    if not vals_list:
        raise ValueError("None of the requested markers were found.")
    df = pd.concat(vals_list, ignore_index=True)
    df = df[np.isfinite(df["value"])]

    first_marker = markers[0]
    order = (
        df.loc[df["marker"] == first_marker]
          .groupby(sample_col)["value"]
          .median()
          .sort_values()
          .index
          .tolist()
    )
    thr_tbl = None
    if add_thresholds and "adt_thresholds" in adata.uns:
        t = adata.uns["adt_thresholds"]
        if isinstance(t, pd.DataFrame):
            t = t.copy()
            t.index = t.index.astype(str)
            thr_tbl = t
    g = sns.FacetGrid(
        df, row=sample_col, col="marker",
        row_order=order,
        col_order=[m for m in markers if m in df["marker"].unique()],
        height=height, aspect=aspect,
        sharex=sharex, sharey=sharey,
        margin_titles=True
    )
    g.map_dataframe(
        sns.histplot,
        x="value",
        bins=bins,
        stat=stat,
        alpha=0.45,
        edgecolor=None
    )

    if kde:
        def _kde_map(data, **kwargs):
            _x = data["value"]
            if drop_zeros_for_kde:
                _x = _x[_x > 0]
            if _x.size >= 10:
                sns.kdeplot(x=_x, bw_adjust=kde_bw_adjust, lw=1.4, **kwargs)

        g.map_dataframe(_kde_map)

    axes = g.axes.flatten() if np.ndim(g.axes) else [g.axes]
    facet_groups = df.groupby([sample_col, "marker"])
    row_labels = g.facet_data()[0].row_names
    col_labels = g.facet_data()[0].col_names
    ax_idx = 0
    for r_name in row_labels:
        for c_name in col_labels:
            if ax_idx >= len(axes):
                break
            ax = axes[ax_idx]
            key = (r_name, c_name)
            if key in facet_groups.groups:
                sub = facet_groups.get_group(key)["value"].values
                sub = sub[np.isfinite(sub)]
                if sub.size > 0:
                    counts, edges = np.histogram(sub, bins=bins)
                    if stat == "density":
                        widths = np.diff(edges)
                        heights = counts / (counts.sum() * widths) if counts.sum() > 0 else counts
                    else:
                        heights = counts
                    ymax = np.quantile(heights[heights > 0], y_cap_quantile) if np.any(heights > 0) else 1.0
                    ax.set_ylim(0, ymax * 1.1)
            ax_idx += 1
    if thr_tbl is not None:
        ax_idx = 0
        for r_name in row_labels:
            for c_name in col_labels:
                if ax_idx >= len(axes):
                    break
                ax = axes[ax_idx]
                if (r_name in thr_tbl.index) and (c_name in thr_tbl.columns):
                    thr = thr_tbl.loc[r_name, c_name]
                    if pd.notna(thr) and np.isfinite(thr):
                        ax.axvline(thr, ls="--", lw=1, alpha=0.9)
                ax_idx += 1

    g.set_axis_labels("value", "Density" if stat == "density" else "Count")
    if title is None:
        title = f"Histogram + KDE (rows={sample_col}, cols=marker)"
    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle(title)
    
    return g

def paged_plot(
    data,
    facet_col="sample_id",
    x=None,
    y=None,
    color=None,
    mark=so.Bar(),
    stat=so.Hist(),
    layout_size=(6, 4),
    ncols=2,
    nrows=2,
    plot_theme=None,
    label_kwargs=None,
    share_axes=True,
    show_slider=True
):
    """
    Interactive paginated faceting with seaborn.objects + ipywidgets.

    Parameters
    ----------
    data : pandas.DataFrame
        Data with columns referenced by `x`, `y`, `color`, and `facet_col`.
    facet_col : str, default: "sample_id"
        Column to facet on (each value becomes a small multiple).
    x, y, color : str or None
        Column names to map to respective aesthetics.
    mark : seaborn.objects.Mark, default: so.Bar()
        Marker type (e.g., so.Bar(), so.Dot(), so.Violin()).
    stat : seaborn.objects.Stat or None, default: so.Hist()
        Aggregation/statistics layer; set None to plot raw marks only.
    layout_size : tuple, default: (6,4)
        Size (inches) per subplot.
    ncols, nrows : int, default: 2,2
        Number of columns and rows per page.
    plot_theme : dict or None
        Theme dict passed to `.theme()`.
    label_kwargs : dict or None
        Dict passed to `.label()`, e.g., {"x":"X label","y":"Count"}.
    share_axes : bool, default: True
        If False, axes can scale independently per facet.
    show_slider : bool, default: True
        Show the page slider widget.

    Returns
    -------
    None
        Displays the interactive plot with a page slider in notebooks.

    Examples
    --------
    >>> paged_plot(
    ...     df, facet_col="sample_id", x="value", y=None,
    ...     mark=so.Hist(), stat=so.Hist(), ncols=3, nrows=2
    ... )
    """
    facet_levels = sorted(data[facet_col].unique())
    per_page = ncols * nrows
    total_pages = (len(facet_levels) + per_page - 1) // per_page

    plot_theme = plot_theme or {
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7
    }
    label_kwargs = label_kwargs or {"x": x or "", "y": y or "Count"}

    def plot_page(page_idx):
        clear_output(wait=True)
        start = page_idx * per_page
        end = start + per_page
        current_facets = facet_levels[start:end]
        page_data = data[data[facet_col].isin(current_facets)]
        p = so.Plot(page_data, x=x, y=y, color=color).facet(col=facet_col, wrap=ncols)
        if stat is not None:
            p = p.add(mark, stat)
        else:
            p = p.add(mark)

        if not share_axes:
            p = p.share(x=False, y=False)

        p = (
            p.layout(size=layout_size)
             .label(**label_kwargs)
             .theme(plot_theme)
        )
        p.show()
        if show_slider:
            display(page_slider)

    if show_slider:
        page_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=total_pages - 1,
            step=1,
            description='Page:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )
        page_slider.observe(lambda change: plot_page(change["new"]) if change["name"] == "value" else None)
    plot_page(0)
