import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import pdb
import csv
import pickle

csv_map = {}

pkl_file = open('../dumps/dump_xception_cutout_aug.pkl', 'rb')
dump0 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_resnet50_cutout_aug.pkl', 'rb')
dump1 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_alexnet_cutout_aug.pkl', 'rb')
dump2 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_squeeze1_1_cutout_aug.pkl', 'rb')
dump3 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_nasnet_cutout_aug.pkl', 'rb')
dump4 = pickle.load(pkl_file)
pkl_file.close()

#pkl_file = open('../dumps/dump_resnet34_aug_nofreeze.pkl', 'rb')
#dump5 = pickle.load(pkl_file)
#pkl_file.close()

for fnum in dump1:
    avg_arr = (dump0[fnum]+dump1[fnum]+dump2[fnum]+dump3[fnum]+dump4[fnum])/5.0
    y_pred = torch.from_numpy(avg_arr)
    smax = nn.Softmax()
    smax_out = smax(y_pred)
    c = np.argmax(smax_out.data).item()
    csv_map[fnum] = c
    #print(fnum, ": ", c)

with open("../submissions/submission_xcep_res50_alex_squ_nas_allcut.csv", 'w') as csvfile:
    fieldnames = ['image_name', 'label']
    csvfile.write('image_name,label')
    csvfile.write('\n')
    for (fname, c) in sorted(csv_map.items()):
        csvfile.write(str(fname)+".jpg,"+str(c))
        csvfile.write("\n")
