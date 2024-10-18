"""
Module for preprocessing and saving scRNA-seq data for trajectory analysis using PCA.

It applies Principal Component Analysis (PCA) to reduce the dimensionality of the dataset, optionally filters the data by specific timesteps, and saves the processed results and corresponding labels for downstream analysis.

Main steps:
    - Load the dataset (in `.npz` format) containing PCA embeddings and sample labels.
    - Standardize (whiten) the PCA embeddings using `StandardScaler`.
    - Select a specified number of PCA components to retain, as provided via command-line arguments.
    - Save the processed PCA-transformed data and sample labels in `.npy` format.

Command-line arguments:
    - ``--n-components``: Number of PCA components to retain (default: 5).

Example usage:
    To run the script with 5 PCA components
    
    .. code-block:: bash
    
        python script.py --n-components 5
"""
import os
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler

def main(n_components: int) -> None:
    """
    Load, preprocess and save scRNA-seq PCA-transformed data for trajectory analysis.

    Parameters
    ----------
    n_components : int
        Number of PCA components to retain.

    Returns
    -------
    None

    References
    ----------
    1. Tong, A., Huang, J., Wolf, G., Van Dijk, D., & Krishnaswamy, S. (2020, November).
       TrajectoryNet: A dynamic optimal transport network for modeling cellular dynamics.
       In International Conference on Machine Learning (pp. 9526-9536). PMLR.
    """
    data_file = os.path.join(".", "data", "TrajectoryNet", "eb_velocity_v5.npz")
    data_dict = np.load(data_file, allow_pickle=True)

    sample_labels = data_dict["sample_labels"]
    pca_embeddings = data_dict["pcs"]
    # Scaling as in https://github.com/KrishnaswamyLab/TrajectoryNet/blob/master/TrajectoryNet/dataset.py
    scaler = StandardScaler()
    scaler.fit(pca_embeddings)
    pca_embeddings = scaler.transform(pca_embeddings)
    n_components = args.n_components
    data = pca_embeddings[:, :n_components]

    folder = f"data/RNA_PCA_{args.n_components}"

    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "data.npy"), data)
    np.save(os.path.join(folder, "sample_labels.npy"), sample_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n-components',
        type=int,
        default=5,
        help=f"""Number of components to keep in PCA.""",
        )

    args = parser.parse_args()
    main(args.n_components)