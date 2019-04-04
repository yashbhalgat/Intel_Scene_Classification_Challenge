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
