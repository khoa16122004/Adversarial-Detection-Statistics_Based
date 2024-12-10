import torch
from PIL import Image
import os
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torchvision.utils import save_image




def get_model():
    model = torch.load("classifier/cnn_brain_pretrained.pt", map_location="cuda:0")
    return model.eval()

def get_dataset(split, take_label):
    testing = False
    
    if split == "test":
        img_folder = r"Image\Brain_Tumor\Testing"
        testing = True

    elif split == "FGSM":
        img_folder = r"Image\Brain_Tumor\AT_FGSM"
        
    elif split == "DDN":
        img_folder = r"Image\Brain_Tumor\AT_DDN"
    
    elif split == "PGD":
        img_folder = r"Image\Brain_Tumor\AT_PGD"
        
    elif split == "flips":
        img_folder = r"Image\Brain_Tumor\Flips"
            
    elif split == "subsampling":
        img_folder = r"Image\Brain_Tumor\Subsampling"
        
    elif split == "gaussian_blur":
        img_folder = r"Image\Brain_Tumor\Gaussian_blur"
    

    dataset = BrainTumorDataset(img_folder, take_label, testing)
    return dataset    

class BrainTumorDataset(Dataset):   
    def __init__(self, img_folder, take_label=False, testing=False,
                 transform=transforms.Compose([transforms.Resize((384,384)), 
                                               transforms.ToTensor()])):
        
        self.img_paths = []
        self.img_names = []
        self.labels = []
        self.transform = transform
        self.take_label = take_label
        self.label_num2str = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
        self.label_str2num = {'glioma': 0, 'meningioma':1, 'notumor':2, 'pituitary':3}

        
        for class_name in os.listdir(img_folder):
            class_folder = os.path.join(img_folder, class_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                self.img_paths.append(file_path)
                self.img_names.append(file_name)
                if take_label:
                    if testing:
                        self.labels.append(self.label_str2num[class_name])
                    else:
                        self.labels.append(int(class_name))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = self.img_names[idx]
        img = Image.open(img_path).convert('RGB')
        
        
        if self.transform:
            img = self.transform(img)
  
        if self.take_label == True:
            label_ts = self.labels[idx]
            return img, img_name, label_ts
        else:
            return img, img_name





# a = get_dataset("test", True)
# save_image(a[0][0], "test_1.png")