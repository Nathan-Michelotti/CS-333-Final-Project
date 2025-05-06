import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

# Paths
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(RESULTS_PATH, exist_ok=True)
GRAPH_PATH = os.path.join(RESULTS_PATH, "graphs")
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)

EMBEDDINGS_PATH = "/home/nmichelotti/Desktop/ALL/Embeddings/embeddings_for_n8/model_240000_DoppelVer_All_112x112_outputs.pth"
EMBEDDINGS_IMAGE_PATH = "/home/nmichelotti/Desktop/ALL/Embeddings/embeddings_for_n8/model_240000_DoppelVer_All_112x112_image_paths.txt"
CLASS_NUM_NAME_PATH = "/home/nmichelotti/Desktop/ALL/Embeddings/embeddings_for_n8/class_num_name.csv"

device = torch.device("cuda")

def load_embeddings():
    embeddings = pd.DataFrame(torch.load(EMBEDDINGS_PATH, map_location=device).cpu())
    with open(EMBEDDINGS_IMAGE_PATH, 'r') as f:
        image_paths = [line.strip().split('/')[-1] for line in f]
    embeddings = embeddings.iloc[:len(image_paths)].copy()
    embeddings["img_name"] = image_paths
    embeddings["class"] = embeddings["img_name"].apply(lambda x: x.split("+")[0])
    return embeddings

def find_midpoints(embeddings):
    midpoints = []
    for class_name in tqdm(embeddings["class"].unique(), desc="Midpoints"):
        group = embeddings[embeddings["class"] == class_name].iloc[:, :512]
        midpoint = group.mean().values
        max_dist = np.max(np.linalg.norm(group.values - midpoint, axis=1))
        midpoints.append({
            "class": class_name,
            "midpoint": ",".join(map(str, midpoint)),
            "max_distance": max_dist
        })
    df = pd.DataFrame(midpoints)
    df.to_csv(os.path.join(RESULTS_PATH, "Midpoints_Per_Class.csv"), index=False)
    return df

def compute_overlap(midpoint_csv):
    df = pd.read_csv(midpoint_csv)
    results = []
    for i in tqdm(range(len(df)), desc="Overlap"):
        m1 = np.fromstring(df.loc[i, "midpoint"], sep=",")
        r1 = df.loc[i, "max_distance"]
        for j in range(i + 1, len(df)):
            m2 = np.fromstring(df.loc[j, "midpoint"], sep=",")
            r2 = df.loc[j, "max_distance"]
            dist = np.linalg.norm(m1 - m2)
            if dist < (r1 + r2):
                overlap = (1 - dist / (r1 + r2)) * 100
                results.append({
                    "class_1": df.loc[i, "class"],
                    "class_2": df.loc[j, "class"],
                    "overlap_percentage": overlap
                })
    
    overlap_df = pd.DataFrame(results)

    if not overlap_df.empty and "overlap_percentage" in overlap_df.columns:
        overlap_df = overlap_df.sort_values(by="overlap_percentage", ascending=False)

    
    overlap_df.to_csv(os.path.join(RESULTS_PATH, "Hypersphere_Overlap.csv"), index=False)
    return overlap_df


def plot_overlap(overlap_df):
    for cls in tqdm(overlap_df["class_1"].unique(), desc="Plotting Overlap Graphs"):
        subset = overlap_df[overlap_df["class_1"] == cls]
        if subset.empty:
            continue
        plt.figure(figsize=(12, 6))
        plt.plot(subset["overlap_percentage"].values, marker="o")
        plt.title(f"Overlap % for Class {cls}")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.savefig(os.path.join(GRAPH_PATH, f"{cls}_overlap.png"))
        plt.close()

def cluster_all_embeddings(embeddings, n_clusters=10):
    X = embeddings.iloc[:, :512].values
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_scaled)
    embeddings["kmeans_label"] = labels
    embeddings[["img_name", "class", "kmeans_label"]].to_csv(os.path.join(RESULTS_PATH, "KMeans_All.csv"), index=False)

def main():
    embeddings = load_embeddings()
    mid_df = find_midpoints(embeddings)
    overlap_df = compute_overlap(os.path.join(RESULTS_PATH, "Midpoints_Per_Class.csv"))
    plot_overlap(overlap_df)
    cluster_all_embeddings(embeddings)

if __name__ == "__main__":
    main()
