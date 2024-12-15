# Adversarial Example Detection using Maximum Mean Discrepancy (MMD)

## Overview
This project aims to detect adversarial examples in datasets using the Maximum Mean Discrepancy (MMD) test. Adversarial examples are inputs to a machine learning model that are designed to cause the model to misbehave. The MMD test is a statistical test that measures the difference between two distributions.

## Code Structure

* `hypothesis_test.py`: This file contains the implementation of the MMD test and related functions.
* `create_dataset.py`: This file contains functions for creating and manipulating datasets.
* `data_utils.py`: This file contains utility functions for working with data.
* `test.py`: This file contains the main script for running the MMD test on a dataset.

## Functionality

* The project provides a function `test_distribution_difference` that takes in two datasets and returns the MMD value, p-value, and a boolean indicating whether the null hypothesis (i.e., the two distributions are the same) is rejected.
* The project also provides functions for creating and manipulating datasets, including `create_dataset` and `mix_datasets`.
* The `main` function in `test.py` runs the MMD test on a dataset and prints the results.

## Usage

* Run the MMD test: 

    ```python
    python test.py --adversarial_attack <attack_type> --adversarial_percentage <percentage> --test_size <size> --n_interations <iterations> --test <test_number>
    ```

+ `--adversarial_attack`: Loại tấn công (FGSM, PGD, flips, subsampling, gaussian_blur)
+ `--adversarial_percentage`: Tỷ lệ mẫu dữ liệu tấn công (0.0 - 1.0)
+ `--test_size`: Kích thước mẫu dữ liệu thử nghiệm
+ `--n_interations`: Số lần lặp lại thử nghiệm
+ `--test`: Số thử nghiệm

## Requirements

* Python 3.x
* PyTorch
* NumPy
* SciPy
* Argparse

## Notes

* The project assumes that the datasets are stored in a specific format (e.g., PyTorch tensors).
* The project uses the `argparse` library to parse command-line arguments.
