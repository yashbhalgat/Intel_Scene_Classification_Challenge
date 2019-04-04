'''
Intel Scene Classification Challenge 2019

Yash Bhalgat, yashbhalgat95@gmail.com
'''

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pdb
import csv

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

model = models.squeezenet1_1(pretrained=False, num_classes=6)
model.load_state_dict(torch.load('best_model_squeeze1_1_aug_nofreeze_p09_polyak.pt'))
model.cuda()
model.eval()

# model2 = models.squeezenet1_1(pretrained=False, num_classes=6)
# model2.load_state_dict(torch.load("best_model_squeeze1_1_aug_nofreeze.pt"))
# model2.cuda()
# model2.eval()

csv_map = {}

for i, (images, filepath) in enumerate(test_loader):
    filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
    filepath = int(filepath)
    image_var = torch.autograd.Variable(images.cuda(), volatile=True)
    try:
        y_pred = model(image_var)
        # y_pred2 = model2(image_var)
        # y_pred = (y_pred+y_pred2)*0.5
    except:
        pdb.set_trace()
    smax = nn.Softmax()
    smax_out = smax(y_pred)[0]
    c = np.argmax(smax_out.cpu().data).item()
    # pdb.set_trace()
    csv_map[filepath] = c
    print(filepath, ": ", c)


with open("../submission_squeezenet_augp09_polyak.csv", 'w') as csvfile:
    fieldnames = ['image_name', 'label']
    # csv_w = csv.writer(csvfile)
    csvfile.write('image_name, label')
    csvfile.write('\n')
    for (fname, c) in sorted(csv_map.items()):
        csvfile.write(str(fname)+".jpg,"+str(c))
        csvfile.write("\n")

