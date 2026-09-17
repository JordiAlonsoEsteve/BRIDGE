import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import os
import warnings


def store_pca_density_plot(
    dataset_name, pca_result, density, tissue, cell_type, cutoff
):
    os.makedirs(f"results/sc_processing/{dataset_name}", exist_ok=True)

    plt.figure(figsize=(10, 8))
    # Create contour plot for density
    x = pca_result[:, 0]
    y = pca_result[:, 1]
    # Color points: green for accepted (>= cutoff), red for excluded (< cutoff)
    colors = np.where(density >= cutoff, "green", "red")
    plt.scatter(x, y, c=colors, s=50, marker="x", alpha=0.6)
    plt.title(f"PCA Density Plot for {tissue} - {cell_type}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    # Add custom legend
    legend_elements = [
        Patch(facecolor="green", label="Accepted"),
        Patch(facecolor="red", label="Excluded"),
    ]
    plt.legend(handles=legend_elements, loc="upper left")
    plt.savefig(
        f"results/sc_processing/{dataset_name}/{tissue}_{cell_type}_pca_density.png"
    )
    plt.close()


def pseudo_bulk_processing(
    adata,
    dataset_name,
    donor_min_cells=50,
    sample_size=500,
    leiden_neighbors=30,
    leiden_resolution=0.8,
    percentile_cutoff=1,
    presence_threshold=0.1,
):
    # 1. For each tissue-cell type pair we extract the subset of cells
    # Extract tissues
    tissues = adata.obs["tissue"].unique()
    cell_types = adata.obs["cell_type"].unique()

    # iterate over the combinations of tissue and cell type
    pseudo_bulk_profiles = []
    for tissue in tqdm(tissues):
        for cell_type in cell_types:
            # print(f"Processing {tissue} - {cell_type}...")
            subset = adata[
                (adata.obs["tissue"] == tissue) & (adata.obs["cell_type"] == cell_type)
            ]
            if subset.n_obs < donor_min_cells:
                # print(f"No cells found for {tissue} - {cell_type}, skipping.")
                continue
            try:
                # 2. "A PCA density filter was applied to remove outliers by performing PCA on the scVI
                # latent space representations and calculating density estimat"
                # extract the latent space representations
                latent_space = subset.obsm["X_scvi"]
                # perform PCA on the latent space representations
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(latent_space)
                # calculate density estimates using Kernel Density Estimation (KDE)
                kde = gaussian_kde(pca_result.T)
                density = kde.pdf(pca_result.T)
                # filter out cells below the cutoff
                cutoff = np.percentile(
                    density, percentile_cutoff
                )  # Example: keep top 99% density
                # store_pca_density_plot(dataset_name, pca_result, density, tissue, cell_type, cutoff)
                filtered_subset = subset[density >= cutoff]
            except Exception as e:
                filtered_subset = subset
            # 3. We then processed the data at the donor level. For each donor within the
            # tissue-cell type subset, we included only those donors with at least 50 cells to ensure sufficient
            # representation. For donors meeting this criterion, we randomly sampled 500 cells with replacement from
            # the donor's cells to standardize the sample size across donors and reduce computational load
            donor_cell_count = filtered_subset.obs["donor_id"].value_counts()
            valid_donors = donor_cell_count[donor_cell_count >= donor_min_cells].index
            if len(valid_donors) == 0:
                # print(f"No valid donors with at least donor_min_cells cells for {tissue} - {cell_type}, skipping.")
                continue
            for donor in valid_donors:
                donor_cells = filtered_subset[filtered_subset.obs["donor_id"] == donor]
                # remove the warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Obviously now they are not unique! because replace is True!
                    sampled_cells = donor_cells[
                        np.random.choice(
                            donor_cells.obs_names, size=sample_size, replace=True
                        )
                    ]
                    # 4.  Clustering
                    # was performed on the donor-specific data using the Leiden algorithm with 30 nearest neighbors, 50
                    # principal components and resolution 0.8, based on the scVI representations. This allowed us to capture
                    # intra-donor heterogeneity. For each cluster within the donor's data, we aggregated the raw UMI counts to
                    # create pseudo-bulk expression profiles.
                    # So within this subset of celltype and tissue. For each donor we get several clusters within celllxtissue:

                    # So again we take the scVI representations of the sampled cells and perform Leiden clustering
                    try:
                        sc.pp.neighbors(
                            sampled_cells,
                            n_neighbors=leiden_neighbors,
                            use_rep="X_scvi",
                        )
                    except Exception as e:
                        sc.pp.neighbors(sampled_cells, n_neighbors=leiden_neighbors)
                    # Check cluster quality TODO
                    sc.tl.leiden(
                        sampled_cells,
                        resolution=leiden_resolution,
                        flavor="igraph",
                        n_iterations=2,
                        directed=False,
                    )
                    # Now we have clusters for the sampled cells of this donor.
                    # We can aggregate the raw UMI counts for each cluster to create pseudo-bulk expression profiles

                    # But the uniqueness does not matter because we are clustering over the Leiden clusters, so
                    # effectlively ignoring that several rows (cells) have the same name.
                for cluster in sampled_cells.obs["leiden"].unique():
                    cluster_cells = sampled_cells[
                        sampled_cells.obs["leiden"] == cluster
                    ]
                    # raw UMI counts
                    X = cluster_cells.raw.X
                    genes = cluster_cells.raw.var_names
                    pseudo_bulk_profile = pd.Series(
                        np.asarray(X.sum(axis=0)).ravel(), index=genes
                    )
                    # Sum across cells to get pseudo-bulk profile for this cluster
                    # Here you can save the pseudo_bulk_profile for this cluster, donor, tissue, and cell type as needed
                    # For example, you could save it to a file or store it in a dictionary for later use

                    # 5: . This resulted in multiple pseudo-bulk samples per donor,
                    # capturing intra-donor heterogeneity. Each pseudo-bulk sample was then annotated with metadata
                    # including donor ID, tissue type, cell type, age, sex, ethnicity, and a unique cluster ID combining sex,
                    # donor ID, and Leiden cluster label. Genes were filtered to retain those expressed in at least 10% of the
                    # cells within the dataset, ensuring that lowly expressed genes did not skew the analysis

                    # annotation:
                    metadata = {
                        "donor_id": donor,
                        "tissue": tissue,
                        "cell_type": cell_type,
                        "sex": sampled_cells.obs["sex"].iloc[0],
                        "ethnicity": sampled_cells.obs["self_reported_ethnicity"].iloc[
                            0
                        ],
                        "cluster_id": cluster,
                    }
                    # add the pseudo_bulk_profile and metadata to a list for later processing
                    pseudo_bulk_profiles.append((pseudo_bulk_profile, metadata))

    # just concatenate donor_id, tissue, cell_type and leiden cluster to get a unique cluster id for each pseudo-bulk sample and use it as column name
    # Then we can create a DataFrame where columns are the unique cluster ids and rows are genes, and the values are the pseudo-bulk expression profiles
    # assrt that the index is the same for all profiles (i.e., the genes are in the same order)
    assert all(
        (pseudo_bulk_profiles[0][0].index == profile.index).all()
        for profile, _ in pseudo_bulk_profiles
    ), "Gene indices are not the same across profiles"

    pseudo_bulk_df = pd.DataFrame(
        {
            f"{meta['tissue']}_{meta['cell_type']}_{meta['donor_id']}_{meta['cluster_id']}": profile
            for profile, meta in pseudo_bulk_profiles
        }
    )
    print(f"Initial pseudo-bulk DataFrame shape: {pseudo_bulk_df.shape}")
    # transform into CPTT (no need for gene bias correction with UMI counts, because they only count the end!) Divide by the total counts per column
    pseudo_bulk_df = (
        pseudo_bulk_df.div(pseudo_bulk_df.sum(axis=0), axis=1) * 1e4
    )  # CPTT
    # assert sum by column is 1e4
    assert np.allclose(
        pseudo_bulk_df.sum(axis=0), 1e4
    ), "Column sums are not equal to 1e4 after CPTT transformation"

    # now log-transform the data
    pseudo_bulk_df = np.log1p(
        pseudo_bulk_df
    )  # log1p is log(1+x) to avoid issues with zero counts
    # filter genes to retain those expressed in at least presence_threshold% of the samples
    gene_presence = (pseudo_bulk_df > 0).mean(axis=1)

    filtered_genes = gene_presence[gene_presence >= presence_threshold].index
    pseudo_bulk_df = pseudo_bulk_df.loc[filtered_genes]
    # print the shape
    print(f"Final pseudo-bulk DataFrame shape: {pseudo_bulk_df.shape}")
    # save the final pseudo-bulk DataFrame to a CSV file
    pseudo_bulk_df.to_csv(f"data/processed/mRNA/human/{dataset_name}_CPTT.csv")


if __name__ == "__main__":
    adata = sc.read_h5ad(
        "data/raw/rna_seq_human/TabulaSapiens/5a495302-b7cd-4bf9-853e-95627b00bb03.h5ad"
    )
    # set a seed for reproducibility
    np.random.seed(42)
    pseudo_bulk_processing(
        adata,
        dataset_name="TabulaSapiens",
        donor_min_cells=50,
        sample_size=500,
        leiden_neighbors=30,
        leiden_resolution=0.8,
        percentile_cutoff=1,
        presence_threshold=0,
    )
