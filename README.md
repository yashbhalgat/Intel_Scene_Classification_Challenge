# Intel Scene Classification Challenge 

**Author**: Yash Bhalgat | Rank 3rd Public Leaderboard | Rank 6th Private Leaderboard

[Competition Link](https://datahack.analyticsvidhya.com/contest/practice-problem-intel-scene-classification-challe/). You can get the data from here: [Drive link](https://drive.google.com/open?id=1uf8lhLf2kctuz_DNdvLnAmMbdcE7QY1G)

## Requirements:
* Python3.6
* pytorch==1.0.x
* torchvision==0.2.2
* albumentations==0.1.12
* pretrainedmodels==0.7.4
* fastai==0.7.0
* numpy==1.15.4
* matplotlib==2.2.3
* PIL==5.1.0
* tqdm==4.25.0
* pickle==4.0

## Data
When you download the data, you must organize the images into three folders: `train`, `valid` and `test`.
The `train` and `valid` folders must have subfolders corresponding to the class names. The final directory structure
for the `imgs` folder should look like:
```
imgs
├── test
├── train
│   ├── buildings
│   ├── forest
│   ├── glacier
│   ├── mountain
│   ├── sea
│   └── street
└── valid
    ├── buildings
    ├── forest
    ├── glacier
    ├── mountain
    ├── sea
    └── street
```

To extract the `test` images, you can simply parse the `test_WyRytb0.csv` file. 

I chose a 80%-20% train-validation split. That means, you should transfer 20% of the files from each subfolder
of the `train` directory to the `valid` directory. To do so, you can use the following command appropriately:
```
shuf -n <num_files> -e train/<class_name>/* | xargs -i mv {} valid/<class_name>/
```
Substitute `num_files` with the number of files (20%) you want to move and `class_name` with one of
`buildings  forest  glacier  mountain  sea  street`.

## Training
There are two scripts which can be used to train the models:
`train_evaluate_scene_classification.py` and `fastai_full.py`.
The first file purely uses `torch` and `torchvision`. The second file exploits the abstractions provided by the `fastai`
library to train the models. You can go through the code for the details of the implementation.
The details can also be found in the submitted report.

To train any model using `train_evaluate_scene_classification.py`, you might have to edit the lines `208-238`. These lines
basically load the pretrained weights and replace the last fully-connected layer to accomodate these 6 classes.

For example, while using the [`xception`](https://github.com/Cadene/pretrained-models.pytorch) network,
the last layer is replaced as follows:
```
num_ftrs = model_ft.last_linear.in_features
model_ft.last_linear = nn.Linear(num_ftrs, NUM_CLASSES)
```
### Hyper-parameters
You can edit the file `fine_tuning_config_file.py` to modify the hyperparameters, as per your usage.

## Evaluation
For evaluation on the test-set, the scripts `dump_output.py` and `create_submission.py` are useful.
- Once a model is trained, `dump_output.py` runs the model on the test images and saves the output logits (output of
the fully connected layer) to a `dump_<model_name>.pkl` file. These dump files are useful while performing
ensembling of different trained models.
- During ensembling, we average the logit outputs of different models and then
use them to get the predicted labels. In case of usine a single model, we just use the logit outputs (which are already 
dumped/saved) to compute the predicted labels. This is done using the `create_submission.py` file.

## Miscellaneous Scripts
1. `create_features.py` - There were some other experiments I performed, like using a SVM classifier on top of a trained 
ResNet (or any other network) used as a featurizer. You can use this script to explore this method further. :)
2. `check_validation.py` - You can use this script to inspect which validation images are being misclassified.
3. `patchwise.py` - This was my attempt to use a patchwise classifier (more like the recent [BagNet](https://openreview.net/pdf?id=SkfMWhAqYQ) paper). Didn't work so well.

Thank you. For any questions about this implementation, feel free to reach out to me at yashbhalgat95@gmail.com
