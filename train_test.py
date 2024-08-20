import unittest
import subprocess
import os
import numpy as np


class TestTrainScript(unittest.TestCase):
    def test_jkonet_star_potential_synthetic_data(self):
        # Run the train.py script
        subprocess.run(['python', 'train.py',
                        '--solver', 'jkonet-star-potential',
                        '--dataset', 'potential_styblinski_tang_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_1000_gmm_10_seed_0'],
                       check=True)

        # Read the loss value from the file
        with open('loss.txt', 'r') as f:
            loss = float(f.read().strip())

        # Assert that the loss is within the desired range
        self.assertTrue(0.01 <= loss <= 0.1, f"Loss {loss} is not within the range 0.01-0.1")

    def test_jkonet_star_linear_potential_synthetic_data(self):
        # Run the train.py script
        subprocess.run(['python', 'train.py',
                        '--solver', 'jkonet-star-linear-potential',
                        '--dataset', 'potential_styblinski_tang_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_1000_gmm_10_seed_0'],
                       check=True)

        # Read the loss value from the file
        with open('loss.txt', 'r') as f:
            loss = float(f.read().strip())

        # Assert that the loss is within the desired range
        self.assertTrue(0.01 <= loss <= 0.1, f"Loss {loss} is not within the range 0.01-0.1")

    def test_jkonet_star_time_potential_rna(self):
        # Run the train.py script
        subprocess.run(['python', 'train.py',
                        '--solver', 'jkonet-star-time-potential',
                        '--dataset', 'RNA_PCA_5'],
                       check=True)

        # Read the loss value from the file
        with open('loss.txt', 'r') as f:
            loss = float(f.read().strip())

        # Assert that the loss is within the desired range
        self.assertTrue(0.01 <= loss <= 0.1, f"Loss {loss} is not within the range 0.01-0.1")

    def test_preprocessing_rna(self):
        # Run the train.py script
        subprocess.run(['python', 'preprocess_rna-seq.py',
                        '--n-components', '5'],
                       check=True)

        # Load the processed data
        folder = f"data/RNA_PCA_5"
        data = np.load(os.path.join(folder, "data.npy"))
        sample_labels = np.load(os.path.join(folder, "sample_labels.npy"))

        # Assert the shape of the processed data
        expected_data_shape = (16819, 5)
        self.assertEqual(data.shape, expected_data_shape,
                         f"Data shape {data.shape} does not match expected shape {expected_data_shape}")

        # Assert the shape of the sample labels
        expected_labels_shape = (16819,)
        self.assertEqual(sample_labels.shape, expected_labels_shape,
                         f"Sample labels shape {sample_labels.shape} does not match expected shape {expected_labels_shape}")

    def test_data_generator_rna(self):
        # Run the train.py script
        subprocess.run(['python', 'data_generator.py',
                        '--load-from-file', 'RNA_PCA_5'],
                       check=True)

        expected_shapes = {
            "couplings_train_0_to_1.npy": (3733, 12),
            "couplings_train_1_to_2.npy": (4222, 12),
            "couplings_train_2_to_3.npy": (3965, 12),
            "couplings_train_3_to_4.npy": (4004, 12),
        }
        folder = f"data/RNA_PCA_5"
        # Iterate over each file and check its shape
        for filename, expected_shape in expected_shapes.items():
            filepath = os.path.join(folder, filename)
            data = np.load(filepath)
            self.assertEqual(data.shape, expected_shape,
                             f"Shape of {filename} {data.shape} does not match expected shape {expected_shape}")


if __name__ == '__main__':
    unittest.main()