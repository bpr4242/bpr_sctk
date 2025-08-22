import pandas as pd
import scanpy as sc
from typing import Optional

def is_outlier(adata, metric, nmads=5):
    """
    Mark cells as outliers by MAD thresholding.

    Parameters
    ----------
    adata : AnnData
        Input data.
    metric : str
        Column in `.obs` to test.
    nmads : int
        Number of MADs.

    Returns
    -------
    np.ndarray (bool)
        Mask of outlier cells.
    """
    x = adata.obs[metric].values
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x < med - nmads * mad) | (x > med + nmads * mad)


def run_scrublet_scores(i: int, rna, force: bool = False):
    """
    Run Scrublet doublet detection.

    Parameters
    ----------
    i : int
        Sample index.
    rna : AnnData
        RNA data.
    force : bool
        Re-run even if results exist.

    Returns
    -------
    tuple (int, DataFrame, dict)
        Index, scores dataframe, scrublet results.
    """
    try:
        if "scrublet" in rna.uns and not force:
            return i, rna.obs[["doublet_score", "predicted_doublet"]], rna.uns["scrublet"]

        scrub = sc.external.pp.scrublet(rna, verbose=False)
        scores = rna.obs[["doublet_score", "predicted_doublet"]].copy()
        return i, scores, rna.uns["scrublet"]
    except Exception as e:
        print(f"Scrublet failed: {e}")
        return i, None, None


def plot_qc_summary(adata, output_file=None):
    """
    Plot QC summary (genes, counts, MT%).

    Parameters
    ----------
    adata : AnnData
        Input data.
    output_file : str or None
        Save to file.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].hist(adata.obs["n_genes_by_counts"], bins=50)
    axs[0].set_title("Genes per cell")
    axs[1].hist(adata.obs["total_counts"], bins=50)
    axs[1].set_title("Counts per cell")
    axs[2].hist(adata.obs["pct_counts_mt"], bins=50)
    axs[2].set_title("MT%")
    if output_file:
        fig.savefig(output_file)
    else:
        plt.show()
