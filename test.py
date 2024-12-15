import argparse
from data_utils import get_model, get_dataset
import random
from hypothesis_test import test_distribution_difference

def create_dataset(testing_dataset, adversarial_dataset, 
                   adversarial_percentage, size):
    
    size = len(adversarial_dataset)
    adversarial_size = int(size * adversarial_percentage)
    testing_size = size - adversarial_size

    adversarial_samples = random.sample(adversarial_dataset, adversarial_size)
    
    testing_samples = random.sample(testing_dataset, testing_size)

    combined_dataset = adversarial_samples + testing_samples

    random.shuffle(combined_dataset)

    return combined_dataset
    

def mix_datasets(datasets, proportions):
    if len(datasets) != len(proportions):
        raise ValueError("Số dataset và tỷ lệ phải trùng nhau.")
    if not abs(sum(proportions) - 1.0) < 1e-6:
        raise ValueError("Tổng các tỷ lệ phải bằng 1.")

    mixed_dataset = []

    for dataset, proportion in zip(datasets, proportions):
        sample_size = int(args.test_size * proportion)
        a = random.sample(dataset, sample_size)
        mixed_dataset.extend(random.sample(dataset, sample_size))

    random.shuffle(mixed_dataset)
    return mixed_dataset


def main(args):
    testing_dataset = get_dataset(split="test", take_label=False)
    adversarial_dataset = get_dataset(split=args.adversarial_attack, take_label=False)
    # fliped_imgs = get_dataset(split="flips", take_label=False)
    # sampling_dataset = get_dataset(split="subsampling", take_label=False)

    testing_imgs = [item[0] for item in testing_dataset]  
    adversarial_imgs = [item[0] for item in adversarial_dataset]
    # fliped_imgs = [item[0] for item in fliped_imgs]
    # sampling_imgs = [item[0] for item in sampling_dataset]
    
    # mixed_imgs = mix_datasets([fliped_imgs, sampling_imgs], 
                                    #  [0.05, 0.95]) 
    
    model = get_model()
    

    acc = 0
    for adversarial_percentage in range(10):
        new_imgs = create_dataset(testing_imgs, adversarial_imgs, 
                                  adversarial_percentage * 0.1, args.test_size)
        mmd, p_value, reject = test_distribution_difference(testing_imgs, 
                                                            new_imgs,
                                                            "features", 
                                                            model,
                                                            args.n_interations,
                                                            args.test)

        
        if reject == True:
            acc += 1
            
        # output_path = f"experiment/{args.test}_{args.n_interations}_{args.adversarial_attack}_{args.test_size}_{adversarial_percentage * 0.1}.txt"
        # with open(output_path, "w+") as f:
        #     f.write(f"MMD value: {mmd:.6f}\n")
        #     f.write(f"p-value: {p_value:.6f}\n")
        #     f.write(f"Reject H0 (distributions are different): {reject}\n")

        # break
    
    print(f"Accuracy with {args.test_size}", acc / 9)    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--adversarial_attack',
        choices=['FGSM', "PGD", "flips", "subsampling", "gaussian_blur"],
        default='FGSM',
        type=str,
    )
    
    parser.add_argument(
        '--adversarial_percentage',
        default=0.5,
        type=float,
    )
    
    parser.add_argument(
        "--test_size",
        default=1300,
        type=int,
    )
    
    parser.add_argument(
        "--n_interations",
        default=10000,
        type=int,
    )
    
    parser.add_argument(
        "--test",
        default=1,
        type=int
    )
    
    
    args = parser.parse_args()
    main(args)