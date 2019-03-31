from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

PATH = "../imgs_resized/"
sz = 224

arch = resnet18
bs = 32

data = ImageClassifierData.from_paths(PATH, bs=bs, tfms=tfms_from_model(arch, sz), test_name="test")
learn = ConvLearner.pretrained(arch, data, precompute=True)

def accuracy_np(preds, targs):
    preds, targs = preds.cpu(), targs.cpu()
    preds = np.argmax(preds, 1)
    return (preds.numpy()==targs.numpy()).mean()

# aug_tfms = [RandomLighting(b=0.5, c=0, tfm_y=TfmType.NO),
#            RandomZoom(zoom_max=1),
#            RandomStretch(max_stretch=2),
#            RandomFlip(),
#            GoogleNetResize(targ_sz=224),
#            Cutout(n_holes=8, length=12, tfm_y=TfmType.NO)]

aug_tfms = transforms_side_on+[Cutout(n_holes=8, length=12, tfm_y=TfmType.NO)]

tfms = tfms_from_model(arch, sz, aug_tfms=aug_tfms, max_zoom=1.1)

data = ImageClassifierData.from_paths(PATH, bs=bs, tfms=tfms, test_name="test")
learn = ConvLearner.pretrained(arch, data, precompute=True)

lrf = learn.lr_find()
ind = np.argmin(learn.sched.losses)
lr = learn.sched.lrs[ind-100]

learn.fit(lr, 2, metrics=[accuracy_np])

learn.precompute=False
learn.fit(0.002, 3, cycle_len=1, metrics=[accuracy_np])

learn.unfreeze()
lrs = np.array([lr*0.01, lr*0.1, lr])
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2, metrics=[accuracy_np])
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2, metrics=[accuracy_np])
lrs = lrs*0.5
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2, metrics=[accuracy_np])
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2, metrics=[accuracy_np])
lrs = lrs*0.7
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2, metrics=[accuracy_np])
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2, metrics=[accuracy_np])

#### TESTING
log_preds,y = learn.TTA(is_test=True)
probs = np.mean(np.exp(log_preds),0)
csv_map = {}

import os
filenames = os.listdir('../imgs_resized/test/')

for i,fname in enumerate(filenames):
    csv_map[int(fname[:-4])] = probs[i]

output = open("../dumps/dump_fastai_res18_tanvi.pkl", 'wb')
pickle.dump(csv_map, output)
output.close()

preds = np.argmax(probs, axis=1)
submission = pd.DataFrame(
    {'image_name': filenames,
     'label': preds,
    })
submission.to_csv('../submissions/submission_fastai_res18_tanvi.csv')
