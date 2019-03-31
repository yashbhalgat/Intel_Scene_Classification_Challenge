### Section 1 - First, let's import everything we will be needing.

from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from fine_tuning_config_file import *
import pdb
from tqdm import tqdm

### Non-deeplearning
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score

use_gpu = GPU_MODE
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)

count=0

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



data_dir = DATA_DIR
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=25)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes


### SECTION 3 : Writing the functions that do training and validation phase. 
def create_features(model, phase="train"):
    model.eval()
    
    for i, data in tqdm(enumerate(dset_loaders[phase])):
        inputs, labels = data
        if use_gpu:
            inputs = Variable(inputs.float().cuda())
        else:
            print("Use a GPU!")
        
        features_var = model(inputs)
        features = features_var.squeeze().cpu().data.numpy()

        if i==0:
            X = features
            Y = labels.numpy()
        else:
            X = np.concatenate((X, features), axis=0)
            Y = np.concatenate((Y, labels), axis=0)

    return X, Y



model_ft = models.resnet18(pretrained=False, num_classes=6)
model_ft.load_state_dict(torch.load('best_model_resnet18_aug.pt'))

my_model = nn.Sequential(*list(model_ft.children())[:-1])
for param in my_model.parameters():
    param.requires_grad = False

if use_gpu:
    model_ft.cuda()

train_x, train_y = create_features(my_model, phase="train")
val_x, val_y = create_features(my_model, phase="val")

pdb.set_trace()

clf = NuSVC(gamma='scale', verbose=True)
clf.fit(train_x, train_y)
predictions = clf.predict(val_x)
print("Val accuracy:", accuracy_score(val_y, predictions))
