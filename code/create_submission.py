import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import pdb
import csv
import pickle

csv_map = {}

pkl_file = open('../dumps/dump_fastai_incepres2_full.pkl', 'rb')
dump0 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_fastai_wrn_full.pkl', 'rb')
dump1 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_nasnet_cutout_aug.pkl', 'rb')
dump2 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_resnext101_32_cutout_aug.pkl', 'rb')
dump3 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_dense161_cutout_aug.pkl', 'rb')
dump4 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_fastai_res152_v2_full.pkl', 'rb')
dump5 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_fastai_resnext10164_full.pkl', 'rb')
dump6 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_fastai_res152.pkl', 'rb')
dump7 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_fastai_incep4_full.pkl', 'rb')
dump8 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_fastai_dn161.pkl', 'rb')
dump9 = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('../dumps/dump_fastai_res152_full.pkl', 'rb')
dump10 = pickle.load(pkl_file)
pkl_file.close()

for fnum in dump0:
    #avg_arr = (dump0[fnum]+dump1[fnum]+2*dump2[fnum]+dump3[fnum]+6*dump4[fnum]+dump5[fnum]+dump6[fnum]+dump7[fnum]+dump8[fnum]+6*dump9[fnum])/13.0
    avg_arr = (dump4[fnum]+dump5[fnum]+dump6[fnum]+dump7[fnum]+dump8[fnum]+dump9[fnum]+dump10[fnum])/6.0
    #avg_arr = dump0[fnum]
    y_pred = torch.from_numpy(avg_arr)
    smax = nn.Softmax()
    smax_out = smax(y_pred)
    c = np.argmax(smax_out.data).item()
    csv_map[fnum] = c
    #print(fnum, ": ", c)

#with open("../submissions/submission_xcep_wrn_full_2_nas_resnext32_6_dense161_res152_v2_full_resnext64_full_res152_incep4_v2_full_fast_3_dn161.csv", 'w') as csvfile:
with open("../submissions/submission_dense161_cut_res152_v2_full_resnext64_full_res152_incep4_full_fast_dn161_res152_full.csv", 'w') as csvfile:
#with open("../submissions/submission_xcep_cutout_full.csv", 'w') as csvfile:
    fieldnames = ['image_name', 'label']
    csvfile.write('image_name,label')
    csvfile.write('\n')
    for (fname, c) in sorted(csv_map.items()):
        csvfile.write(str(fname)+".jpg,"+str(c))
        csvfile.write("\n")
