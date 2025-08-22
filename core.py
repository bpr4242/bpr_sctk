import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from typing import List

def remove_elements(original_list: List[str], elements_to_remove: List[str]) -> List[str]:
    """
    Remove elements from a list.

    Parameters
    ----------
    original_list : list of str
        The original list.
    elements_to_remove : list of str
        Elements to remove.

    Returns
    -------
    list of str
        New list with elements removed.
    """
    return [el for el in original_list if el not in elements_to_remove]


def generate_processed_PAGA_UMAP(adata, basis="X_pca", neighbors_key="neighbors", key_added="X_umap_paga", random_state=42):
    """
    Compute UMAP embedding initialized with PAGA positions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    basis : str, default="X_pca"
        Embedding to use for neighbors.
    neighbors_key : str, default="neighbors"
        Key for precomputed neighbor graph.
    key_added : str, default="X_umap_paga"
        Key under `.obsm` to store new embedding.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    AnnData
        Updated with `.obsm[key_added]`.
    """
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos="paga", neighbors_key=neighbors_key, random_state=random_state)
    adata.obsm[key_added] = adata.obsm["X_umap"]
    return adata


def clr_prot(adata, inplace=True, layer=None):
    """
    Apply centered log-ratio (CLR) normalization to protein modality.

    Parameters
    ----------
    adata : AnnData
        Protein AnnData.
    inplace : bool, default=True
        If True, updates `.X` (or `layer`).
    layer : str or None
        If set, write result to `adata.layers[layer]`.

    Returns
    -------
    AnnData or np.ndarray
        Normalized data if `inplace=False`.
    """
    import scipy
    import numpy as np
    X = adata.X if not layer else adata.layers[layer]
    X = np.asarray(X)
    gm = np.exp(np.mean(np.log1p(X), axis=1))[:, None]
    clr = np.log1p(X / gm)
    if inplace:
        if layer:
            adata.layers[layer] = clr
        else:
            adata.X = clr
        return adata
    else:
        return clr
