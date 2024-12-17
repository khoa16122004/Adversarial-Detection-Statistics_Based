import argparse
import torch
import numpy as np
import random
from data_utils import get_model, get_dataset
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from tqdm import tqdm

def gaussian_kernel(x, y, sigma=1.0):
    
    # ex(- ||x - y||^2 / (2sigma^2))
    
    x_norm = x.pow(2).sum(1).view(-1, 1)
    y_norm = y.pow(2).sum(1).view(-1, 1)
    dist = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t()) 
    return torch.exp(-dist / (2 * sigma * sigma))

def compute_mmd(x, y, kernel_function=gaussian_kernel):
    n = x.size(0)
    m = y.size(0)
    
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.FloatTensor(y)
    
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    
    k_xx = kernel_function(x, x)
    k_yy = kernel_function(y, y)
    k_xy = kernel_function(x, y)
    
    mmd = (k_xx.sum() / (n * n) + 
           k_yy.sum() / (m * m) - 
           2 * k_xy.sum() / (n * m))
    
    return mmd.item()

def bootstrap_mmd_test(x, y, n_iterations=10000, test=1, alpha=0.05):
    
    """
    - if two dataser has the same distribution, the diff > original is random things. If pvalue < alpha, the random is small and this dont have statitical meaning -> reject H0. In contrast if p_value > alpha, the random is high and this have significant statical meaning -> accept H0. \
    - Hypothesis test:    
    H0: distributions are identical
    H1: distributions are different
    
    Returns:
    - MMD value
    - p-value
    - boolean indicating if H0 should be rejected
    """
    n = len(x)
    m = len(y)
    combined = np.vstack([x, y])
    
    original_mmd = compute_mmd(torch.FloatTensor(x), torch.FloatTensor(y))
    if test == 0:
        return original_mmd, 0, 0
    
    bootstrap_mmds = []
    for _ in tqdm(range(n_iterations), desc="Bootstrapping"):
        perm = np.random.permutation(n + m)
        bootstrap_x = combined[perm[:n]]
        bootstrap_y = combined[perm[n:]]
        
        bootstrap_mmd = compute_mmd(torch.FloatTensor(bootstrap_x), 
                                  torch.FloatTensor(bootstrap_y))
        bootstrap_mmds.append(bootstrap_mmd)
    
    p_value = np.mean(np.array(bootstrap_mmds) >= original_mmd)
    
    # error means: resampling > observed
    
    if p_value > alpha: # more than alpha percent likelihood that this error is random
        reject_h0 = False
    else:
        reject_h0 = True # p_value percent likelihood the error is random, 1 - pvalue two distribution are different
    
    # reject_h0 = p_value < alpha
    
    return original_mmd, p_value, reject_h0

def sampling_dataset(dataset_1, dataset_2, sampling_percentage=0.5):
    sample_size_1 = int(len(dataset_1) * (1 - sampling_percentage))
    sample_size_2 = int(len(dataset_2) * sampling_percentage)
    
    sampled_dataset_1 = random.sample(dataset_1, sample_size_1)
    sampled_dataset_2 = random.sample(dataset_2, sample_size_2)
    
    new_dataset = sampled_dataset_1 + sampled_dataset_2
     
    return new_dataset    

def extract_features_with_sigmoid(model, imgs):
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    sigmoid = nn.Sigmoid()
    
    features = []
    for img in imgs:
        with torch.no_grad():
            feat = feature_extractor(img.unsqueeze(0).cuda())
            feat_sigmoid = sigmoid(feat).cpu().view(-1).numpy()
            features.append(feat_sigmoid)
    
    return np.array(features)

def test_distribution_difference(x, y, method="feautures", model=None, n_iterations=10000, test=0):
    if method == "images":
        x_flat = torch.stack([img.view(-1) for img in x]).cpu().numpy()
        y_flat = torch.stack([img.view(-1) for img in y]).cpu().numpy()
        
        mmd, p_value, reject = bootstrap_mmd_test(x_flat, y_flat)
        
    elif method == "features":
        features_x = extract_features_with_sigmoid(model, x)
        features_y = extract_features_with_sigmoid(model, y)
        
        # scaler = MinMaxScaler()
        # features_x_scaled = scaler.fit_transform(features_x)
        # features_y_scaled = scaler.transform(features_y)
        
        mmd, p_value, reject = bootstrap_mmd_test(features_x, features_y, n_iterations, test)
    
    return mmd, p_value, reject, features_x, features_y

# def main(args):
#     testing_dataset = get_dataset(split="test", take_label=False)
    
#     for method in ['FGSM', "DDN", "PGD", "flips", "subsampling", "gaussian_blur"]:
#         try:
#             print(f"\nMethod: {method}")
#             transform_dataset = get_dataset(split=method, take_label=False)

#             testing_imgs = [item[0] for item in testing_dataset]  
#             transform_imgs = [item[0] for item in transform_dataset]
#             new_sample_dataset = sampling_dataset(testing_imgs, transform_imgs)
            
#             model = get_model() if args.method == "features" else None
            
#             mmd, p_value, reject = test_distribution_difference(
#                 testing_imgs, 
#                 new_sample_dataset, 
#                 method=args.method,
#                 model=model
#             )
            
#             print(f"MMD value: {mmd:.6f}")
#             print(f"p-value: {p_value:.6f}")
#             print(f"Reject H0 (distributions are different): {reject}")
                
#         except Exception as e:
#             print(f"Error with method {method}: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Test distribution differences using MMD.")
    
#     parser.add_argument(
#         '--method',
#         choices=['images', 'features'],
#         default='images',
#         help="Select the method for computing MMD ('images' or 'features'). Default is 'images'."
#     )
    
#     args = parser.parse_args()
#     main(args)