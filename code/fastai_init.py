from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "../imgs_resized/"
sz = 224

arch = wrn

data = ImageClassifierData.from_paths(PATH, bs=64, tfms=tfms_from_model(arch, sz), test_name="test")
learn = ConvLearner.pretrained(arch, data, precompute=True)

def accuracy_np(preds, targs):
    preds, targs = preds.cpu(), targs.cpu()
    preds = np.argmax(preds, 1)
    return (preds.numpy()==targs.numpy()).mean()

aug_tfms = [RandomLighting(b=0.5, c=0, tfm_y=TfmType.NO),
            RandomRotateZoom(deg=25, zoom=2, stretch=1),
            RandomStretch(max_stretch=2),
            Cutout(n_holes=5, length=16, tfm_y=TfmType.NO)]

tfms = tfms_from_model(arch, sz, aug_tfms=aug_tfms, max_zoom=1.1)
