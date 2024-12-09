import argparse
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import gaussian_blur
import os
from data_utils import get_model, get_dataset
from tqdm import tqdm
from torchvision.utils import save_image


def flip(image):
    return image.flip(-1)

def subsample(image, scale=0.5):
    _, h, w = image.size()
    new_h, new_w = int(h * scale), int(w * scale)
    return torch.nn.functional.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear').squeeze(0)

def gaussian_blur_transform(image, kernel_size=3):
    return gaussian_blur(image, kernel_size=[kernel_size, kernel_size])

def apply_transforms(dataset, transform_type):
    transformed_dataset = []
    for image, img_name, label in dataset:
        if transform_type == 'flips':
            transformed_image = flip(image)
        elif transform_type == 'subsampling':
            transformed_image = subsample(image)
        elif transform_type == 'gaussian_blur':
            transformed_image = gaussian_blur_transform(image)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        transformed_dataset.append((transformed_image, img_name, label))
    return transformed_dataset

def main():
    parser = argparse.ArgumentParser(description="Apply transformations to a dataset")
    parser.add_argument('--transform', type=str, required=True, 
                        choices=['flips', 'subsampling', 'gaussian_blur'],
                        help="Type of transformation to apply")
    parser.add_argument('--output_folder', type=str, required=True, 
                        help="Folder to save transformed dataset")
    args = parser.parse_args()

    model = get_model()
    original_dataset = get_dataset(split="test", take_label=True)

    transformed_dataset = apply_transforms(original_dataset, args.transform)

    for idx, (image, img_name, label) in tqdm(enumerate(transformed_dataset)):
        label_str = str(label) if isinstance(label, int) else label
        
        label_dir = os.path.join(args.output_folder, label_str)
        os.makedirs(label_dir, exist_ok=True)
        
        output_path = os.path.join(label_dir, img_name)
        save_image(image, output_path)

if __name__ == '__main__':
    main()
