import sys
import os
import unittest
import pandas as pd
import numpy as np
import torch

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/HyperShperes")))

from celeb_overlap_analysis import (
    find_midpoints,
    load_embeddings,
    compute_overlap,
    cluster_all_embeddings,
)

# Auto-generate required test files
def generate_dummy_test_files():
    os.makedirs("tests", exist_ok=True)

    dummy_embeddings = [torch.randn(512) for _ in range(5)]
    torch.save(dummy_embeddings, "tests/test_embeddings.pth")

    with open("tests/test_image_paths.txt", "w") as f:
        f.writelines(["A+1.png\n", "A+2.png\n", "B+1.png\n", "C+1.png\n", "C+2.png\n"])

    df = pd.DataFrame({
        "class": ["A", "B"],
        "midpoint": [
            ",".join(map(str, np.zeros(512))),
            ",".join(map(str, np.ones(512) * 0.5))
        ],
        "max_distance": [1.0, 1.0]
    })
    df.to_csv("tests/test_midpoints.csv", index=False)

generate_dummy_test_files() 

class TestOverlapAnalysis(unittest.TestCase):

    def test_find_midpoints_output(self):
        dummy_data = pd.DataFrame({
            **{i: [0.1 * i] * 3 for i in range(512)},
            "img_name": ["A+1.png", "A+2.png", "B+1.png"],
            "class": ["A", "A", "B"]
        })
        result = find_midpoints(dummy_data)
        self.assertIn("midpoint", result.columns)
        self.assertIn("max_distance", result.columns)
        self.assertEqual(len(result), 2)

    def test_load_embeddings_basic(self):
        embeddings = pd.DataFrame(torch.load("tests/test_embeddings.pth"))
        with open("tests/test_image_paths.txt", "r") as f:
            names = [line.strip() for line in f]
        embeddings = embeddings.iloc[:len(names)].copy()
        embeddings["img_name"] = names
        embeddings["class"] = embeddings["img_name"].apply(lambda x: x.split("+")[0])
        self.assertEqual(len(embeddings), 5)
        self.assertIn("class", embeddings.columns)

    def test_compute_overlap_runs(self):
        df = compute_overlap("tests/test_midpoints.csv")
        self.assertIsInstance(df, pd.DataFrame)

    def test_cluster_all_embeddings_runs(self):
        dummy_data = pd.DataFrame({
            **{i: np.random.rand(10) for i in range(512)},
            "img_name": [f"img_{i}.png" for i in range(10)],
            "class": ["A"] * 10
        })
        cluster_all_embeddings(dummy_data, n_clusters=2)
        out_path = os.path.join(os.path.dirname(__file__), "../src/results/KMeans_All.csv")
        self.assertTrue(os.path.exists(out_path))
        out = pd.read_csv(out_path)
        self.assertIn("kmeans_label", out.columns)

    def test_load_embeddings_returns_dataframe(self):
        df = load_embeddings()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue("img_name" in df.columns)
        self.assertTrue("class" in df.columns)


if __name__ == "__main__":
    unittest.main()
