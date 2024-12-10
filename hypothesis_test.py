import argparse
import torch
import numpy as np
import random
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from data_utils import get_model, get_dataset
from sklearn.preprocessing import StandardScaler

def sampling_dataset(dataset_1, dataset_2, sampling_percentage=0.9):
    sample_size_1 = int(len(dataset_1) * (1 - sampling_percentage))
    sample_size_2 = int(len(dataset_2) * sampling_percentage)
    
    sampled_dataset_1 = random.sample(dataset_1, sample_size_1)
    sampled_dataset_2 = random.sample(dataset_2, sample_size_2)
    
    new_dataset = sampled_dataset_1 + sampled_dataset_2
    
    return new_dataset    

def compute_kl_divergence_images(dataset_1, dataset_2, bins=256):
    pixels_1 = torch.cat([img.view(-1) for img in dataset_1])
    pixels_2 = torch.cat([img.view(-1) for img in dataset_2])
        
    hist_1, _ = np.histogram(pixels_1.numpy(), bins=bins, range=(0, 1), density=True)
    hist_2, _ = np.histogram(pixels_2.numpy(), bins=bins, range=(0, 1), density=True)    
    
    hist_1 += 1e-8
    hist_2 += 1e-8
    
    kl_div_1_to_2 = entropy(hist_1, hist_2)
    kl_div_2_to_1 = entropy(hist_2, hist_1)
    
    return kl_div_1_to_2, kl_div_2_to_1

def compute_kl_divergence_features(dataset_1, dataset_2, feature_extractor, bandwidth=0.01):
    # Extract features
    features_1 = [feature_extractor(img.unsqueeze(0).cuda()).cpu().view(-1).detach().numpy() for img in dataset_1]
    features_2 = [feature_extractor(img.unsqueeze(0).cuda()).cpu().view(-1).detach().numpy() for img in dataset_2]
    
    features_1 = np.vstack(features_1)
    features_2 = np.vstack(features_2)
    
    # Normalize the features
    scaler = StandardScaler()
    features_1 = scaler.fit_transform(features_1)
    features_2 = scaler.fit_transform(features_2)
    
    # Kernel Density Estimation for feature distributions
    kde_1 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_2 = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    
    kde_1.fit(features_1)
    kde_2.fit(features_2)
    
    # Sample points for KL divergence estimation
    sample_size = min(len(features_1), len(features_2), 1000)
    samples_1 = features_1[:sample_size]
    samples_2 = features_2[:sample_size]
    
    # Compute log probabilities
    log_prob_1_in_1 = kde_1.score_samples(samples_1)
    log_prob_1_in_2 = kde_2.score_samples(samples_1)
    
    log_prob_2_in_2 = kde_2.score_samples(samples_2)
    log_prob_2_in_1 = kde_1.score_samples(samples_2)
    
    # Compute KL divergence
    kl_div_1_to_2 = np.mean(log_prob_1_in_1 - log_prob_1_in_2)
    kl_div_2_to_1 = np.mean(log_prob_2_in_2 - log_prob_2_in_1)
    
    return kl_div_1_to_2, kl_div_2_to_1

def main(args):
    testing_dataset = get_dataset(split="test", take_label=False)  # return img, img_name, label_ts
    
    for method in ['FGSM', "DDN", "PGD", "flips", "subsampling", "gaussian_blur"]:
        try:
            print(f"Method: {method}")
            transform_dataset = get_dataset(split=method, take_label=False)  # img, img_name

            testing_imgs = [item[0] for item in testing_dataset]  
            transform_imgs = [item[0] for item in transform_dataset]

            if args.method == "images":
                kl_div_1_to_2, kl_div_2_to_1 = compute_kl_divergence_images(testing_imgs, transform_imgs)
                print(f"KL Divergence from testing to new sampling (images): {kl_div_1_to_2}")
                print(f"KL Divergence from new sampling to testing (images): {kl_div_2_to_1}")
            
            elif args.method == "features":
                model = get_model()
                feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
                kl_div_1_to_2, kl_div_2_to_1 = compute_kl_divergence_features(testing_imgs, transform_imgs, feature_extractor)
                print(f"KL Divergence from testing to new sampling (features): {kl_div_1_to_2}")
                print(f"KL Divergence from new sampling to testing (features): {kl_div_2_to_1}")
    
        except Exception as e:
            print(f"Error with method {method}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute KL Divergence between datasets based on image or feature distribution.")
    
    parser.add_argument(
        '--method',
        choices=['images', 'features'],
        default='images',
        help="Select the method for computing KL Divergence ('images' or 'features'). Default is 'images'."
    )
    
    args = parser.parse_args()
    main(args)
