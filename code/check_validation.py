import torch
import torch.hub
import pretrainedmodels
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pdb
import pickle

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


inp_size = 331
data_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.CenterCrop(inp_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = "../imgs_resized/"

dsets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms) for x in ['val']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=1, shuffle=True, num_workers=25) for x in ['val']}

dset_sizes = {x: len(dsets[x]) for x in ['val']}

#model = models.resnet18(num_classes=6)
#model.load_state_dict(torch.load('fine_tuned_best_model.pt'))

model = pretrainedmodels.xception(num_classes=1000)
num_ftrs = model.last_linear.in_features
model.last_linear = nn.Linear(num_ftrs, 6)
model.load_state_dict(torch.load('best_model_xception_cutout_aug.pt'))

#model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet20', num_classes=6)
#model.load_state_dict(torch.load('best_model_senet20_aug_nofreeze.pt'))

model.cuda()
model.eval()

for data in dset_loaders['val']:
    image, label, path = data
    image_var = torch.autograd.Variable(image.cuda(), volatile=True)
    y_pred = model(image_var)
    smax = nn.Softmax()
    smax_out = smax(y_pred)[0]
    pred = np.argmax(smax_out.cpu().data).item()
    label = label.cpu().data.item()
    
    if pred!=label:
        print("path:", path, "pred:", pred, "label:", label)
