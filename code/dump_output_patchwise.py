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
from patchwise import Quadrant

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
            print(filename)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inp_size = 224
testdir = "../imgs_resized/test/"
test_loader = data.DataLoader(
        TestImageFolder(testdir,
                        transforms.Compose([
                            transforms.Scale(inp_size),
                            transforms.CenterCrop(inp_size),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False)

model = Quadrant()
model.load_state_dict(torch.load('best_model_dense161_cutout_patchwise.pt'))

model.cuda()
model.eval()

csv_map = {}

for i, (images, filepath) in enumerate(test_loader):
    filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
    filepath = int(filepath)
    image_var = torch.autograd.Variable(images.cuda(), volatile=True)
    patch1 = image_var[:,:,0:112, 0:112]
    patch2 = image_var[:,:,0:112, 113:]
    patch3 = image_var[:,:,113:, 0:112]
    patch4 = image_var[:,:,113:, 113:]
    try:
        y_pred = model(patch1, patch2, patch3, patch4)
    except:
        pdb.set_trace()
    pred = y_pred[0].cpu().data.numpy()
    csv_map[filepath] = pred

output = open('../dumps/dump_dense161_cutout_patchwise.pkl', 'wb')
pickle.dump(csv_map, output)
output.close()
