import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Use relative path
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(RESULTS_PATH, exist_ok=True)

def plot_class_radii_histogram(df, output_path=RESULTS_PATH):
    plt.figure(figsize=(10, 6))
    plt.hist(df["max_distance"], bins=30, color="skyblue", edgecolor="black")
    plt.title("Distribution of Class Radii (Max Distance from Midpoint)")
    plt.xlabel("Max Distance")
    plt.ylabel("Number of Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "class_radii_histogram.png"))
    plt.close()

def plot_class_midpoints_pca(df, output_path=RESULTS_PATH):
    df["midpoint_vector"] = df["midpoint"].apply(lambda x: np.fromstring(x, sep=","))
    midpoint_matrix = np.vstack(df["midpoint_vector"].values)
    pca_coords = PCA(n_components=2).fit_transform(midpoint_matrix)
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.7)
    for i, label in enumerate(df["class"]):
        plt.text(pca_coords[i, 0], pca_coords[i, 1], label, fontsize=6)
    plt.title("PCA Projection of Class Midpoints")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "class_midpoints_pca.png"))
    plt.close()

if __name__ == "__main__":
    example_csv = os.path.join(os.path.dirname(__file__), "../../tests/test_midpoints.csv")
    if os.path.exists(example_csv):
        df = pd.read_csv(example_csv)
        plot_class_radii_histogram(df)
        plot_class_midpoints_pca(df)
