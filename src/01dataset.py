import numpy as np
import torch
import random
import albumentations
import pandas as pd
import os
import pdb

import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
PIL.Image.MAX_IMAGE_PIXELS = 933120000


from torch.utils.data import Dataset, DataLoader


class wikiart_dataset(Dataset):
    def __init__(self, data_folder, csv, num_classes, transform=None):
        '''
        csv format: filename
        flag_train: flag to decide training or testing
        '''
        self.dataframe = pd.read_csv(csv)
        self.dataframe.index = range(len(self.dataframe))
        self.data_folder = data_folder
        self.transform = transform
        self.num_classes = num_classes


    def __getitem__(self, index):

        img_rel_name = self.dataframe['filename'][index]
        img_path = os.path.abspath(os.path.join(self.data_folder, img_rel_name))
        img = Image.open(img_path).convert('RGB')
                
        label = torch.tensor(int(self.dataframe['class'][index]))
        label_onehot = torch.nn.functional.one_hot(label, num_classes=self.num_classes)

        if self.transform is not None:
            img = self.transform(img)
        
        return torch.tensor(img, dtype=torch.float), torch.tensor(label_onehot, dtype=torch.float)
        
    def __len__(self):
        return len(self.dataframe)

# image dataset module
# class WikiArtImageDataset(Dataset):
#     def __init__(self, path, labels, tfms=None):
#         self.X = path
#         self.y = labels
 
#         # apply augmentations
#         if tfms == 0: # if validating
#             self.aug = albumentations.Compose([
#                 albumentations.Resize(224, 224, always_apply=True),
#                 albumentations.Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225], always_apply=True)
#             ])
#         else: # if training
#             self.aug = albumentations.Compose([
#                 albumentations.Resize(224, 224, always_apply=True),
#                 albumentations.HorizontalFlip(p=1.0),
#                 albumentations.ShiftScaleRotate(
#                     shift_limit=0.3,
#                     scale_limit=0.3,
#                     rotate_limit=30,
#                     p=1.0
#                 ),
#                 albumentations.Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225], always_apply=True)
#             ])
 
#     def __len__(self):
#         return (len(self.X))
    
#     def __getitem__(self, i):
#         image = Image.open(self.X[i]).convert('RGB')
#         image = self.aug(image=np.array(image))['image']
#         image = np.transpose(image, (2, 0, 1)).astype(np.float32)
#         label = self.y[i]
 
#         return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)
