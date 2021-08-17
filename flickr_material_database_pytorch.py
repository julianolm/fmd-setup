"""
This code is for loading Flickr Material Database (FMD).
The FMD zip file must be in the drive and the drive must be mounted for the code to work.

Each time "get_fmd_random_loaders" is called new train and validation loaders are generated, easing the process for cross validation.

This work was done by Juliano Amadeu
Github: https://github.com/julianolm
LinkedIn: https://www.linkedin.com/in/juliano-amadeu-78735818b/
"""

import torch
from torchvision import datasets, transforms
import numpy as np


# unzip fmd file
# fmd zip file must be in the drive and the drive must have been mounted
!unzip "/content/drive/MyDrive/fmd.zip"


def get_train_test_indices(train_rate=0.9):
    """
    As classes estao divididas no dataset da seguinte forma: (0-99: classe 1),
                                                             (100-199: classe 2),
                                                             ...
                                                             (900-999: classe 10)
    
    Temos que dividir os indices aleatoriamente para treino e validacao, mas lembrando que as 
    proporcoes definidas para treino e validacao devem ser respeitadas entre cada uma das classes.
    """
    train_indices = []
    val_indices = []
    for i in range(10):
        ind = np.arange(i*100, (i+1)*100)
        ind = np.random.permutation(ind)
        
        frontier = int(len(ind)*train_rate)
        ind_t = ind[0:frontier]
        ind_v = ind[frontier:len(ind)]

        train_indices += ind_t.tolist()
        val_indices += ind_v.tolist()

    return (train_indices, val_indices)


def get_fmd_random_loaders(train_rate=0.5, bs_train=8, bs_val=8):
    train_indices, val_indices = get_train_test_indices(train_rate=0.5)

    trainset = torch.utils.data.Subset(dataset_train, train_indices)
    valset = torch.utils.data.Subset(dataset_val, val_indices)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=bs_val, shuffle=False)

    return train_loader, val_loader
    

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Must load two separate datasets, since their tranforms are not the same
data_dir = '/content/image'
dataset_train = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
dataset_val = datasets.ImageFolder(data_dir, transform=data_transforms['val'])

train_loader, val_loader = get_fmd_random_loaders(0.5, 10, 10)

# #testing:
# import matplotlib.pyplot as plt
# imgs, _ = next(iter(train_loader))
# img = imgs[0]
# plt.imshow(img.permute(1, 2, 0))
