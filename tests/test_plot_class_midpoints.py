import sys
import os
import unittest
import pandas as pd
import numpy as np
import shutil

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/HyperShperes")))

from plot_class_midpoints import (
    plot_class_radii_histogram,
    plot_class_midpoints_pca,
)

class TestPlotClassMidpoints(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/test_results"
        os.makedirs(self.test_dir, exist_ok=True)

        # Auto-generate midpoint CSV
        midpoint = ",".join(map(str, np.random.rand(512)))
        self.df = pd.DataFrame({
            "class": ["A", "B"],
            "midpoint": [midpoint, midpoint],
            "max_distance": [1.0, 0.8]
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_plot_class_radii_histogram_runs(self):
        plot_class_radii_histogram(self.df, output_path=self.test_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "class_radii_histogram.png")))

    def test_plot_class_midpoints_pca_runs(self):
        plot_class_midpoints_pca(self.df, output_path=self.test_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "class_midpoints_pca.png")))

if __name__ == "__main__":
    unittest.main()
