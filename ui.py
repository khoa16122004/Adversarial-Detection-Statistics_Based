import streamlit as st
import numpy as np
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_utils import get_dataset, get_model
from hypothesis_test import test_distribution_difference
from test import create_dataset

def main():
    st.title("Distribution Comparison App")

    # Sidebar for inputs
    st.sidebar.header("Hypothesis Test Configuration")
    
    # Dataset selection
    attack_types = ['FGSM', 'PGD', 'flips', 'subsampling', 'gaussian_blur']
    adversarial_attack = st.sidebar.selectbox("Select Adversarial Attack", attack_types)
    
    # Percentage inputs
    adversarial_percentage = st.sidebar.slider(
        "Adversarial Dataset Percentage", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )
    
    # Test configuration
    test_size = st.sidebar.number_input(
        "Test Size", 
        min_value=100, 
        max_value=10000, 
        value=1300
    )
    
    n_iterations = st.sidebar.number_input(
        "Number of Bootstrap Iterations", 
        min_value=100, 
        max_value=50000, 
        value=10000
    )
    
    test_type = st.sidebar.selectbox(
        "Test Type", 
        [1, 0],
        format_func=lambda x: "Bootstrap Test" if x == 1 else "No Test"
    )
    
    alpha = st.sidebar.slider(
        "Significance Level (α)", 
        min_value=0.01, 
        max_value=0.1, 
        value=0.05, 
        step=0.01
    )
    model = get_model()

    # Button to run analysis
    if st.sidebar.button("Run Distribution Comparison"):
        # Placeholder for dataset loading functions
        # You'll need to replace these with your actual data loading methods

        # Load datasets
        testing_dataset = get_dataset(split="test", take_label=False)
        adversarial_dataset = get_dataset(split=adversarial_attack, take_label=False)
        
        # Extract images (this assumes the first element is the image)
        testing_imgs = [item[0] for item in testing_dataset]  
        adversarial_imgs = [item[0] for item in adversarial_dataset]
        
        # Create combined dataset
        new_imgs = create_dataset(testing_imgs, adversarial_imgs, 
                                  adversarial_percentage, test_size)
        
        mmd, p_value, reject, features_x, features_y = test_distribution_difference(testing_imgs, 
                                                            new_imgs,
                                                            "features", 
                                                            model,
                                                            n_iterations,
                                                            True)
        
        # Display results
        st.header("Hypothesis Test Results")
        st.write(f"**MMD Value:** {mmd:.6f}")
        st.write(f"**p-value:** {p_value:.6f}")
        st.write(f"**Reject Null Hypothesis:** {reject}")
        
        # # Visualization of distributions
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title("Original Dataset")
        # plt.hist(np.array(features_x).flatten(), bins=30)
        # plt.subplot(1, 2, 2)
        # plt.title("Modified Dataset")
        # plt.hist(np.array(features_y).flatten(), bins=30)
        # st.pyplot(plt)

if __name__ == "__main__":
    main()