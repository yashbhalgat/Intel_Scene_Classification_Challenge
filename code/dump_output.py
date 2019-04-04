'''
Intel Scene Classification Challenge 2019

Yash Bhalgat, yashbhalgat95@gmail.com
'''

import torch
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

class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if img.layers==1:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

testdir = "../imgs_resized/test/"
test_loader = data.DataLoader(
        TestImageFolder(testdir,
                        transforms.Compose([
                            transforms.Scale(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False)

model = pretrainedmodels.xception(num_classes=1000)
#model = models.squeezenet1_1()

num_ftrs = model.last_linear.in_features
model.last_linear = nn.Linear(num_ftrs, 6)
#model.classifier = nn.Sequential(
#                        nn.Dropout(p=0.5),
#                        nn.Conv2d(512, 6, kernel_size=1),
#                        nn.ReLU(inplace=True),
#                        nn.AdaptiveAvgPool2d((1, 1))
#                        )
#model.num_classes = 6

model.load_state_dict(torch.load('best_model_xcep_cutout_full.pt'))
model.cuda()
model.eval()

csv_map = {}

for i, (images, filepath) in enumerate(test_loader):
    filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
    filepath = int(filepath)
    image_var = torch.autograd.Variable(images.cuda(), volatile=True)
    try:
        y_pred = model(image_var)
    except:
        pdb.set_trace()
    pred = y_pred[0].cpu().data.numpy()
    csv_map[filepath] = pred

output = open('../dumps/dump_xcep_cutout_full.pkl', 'wb')
pickle.dump(csv_map, output)
output.close()
