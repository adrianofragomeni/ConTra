# ConTra
This repo contains code and data of ConTra model used in the ACCV oral paper ConTra: (Con)text (Tra)nsformer for Cross-Modal Video Retrieval.

## Features
The video features of ActivityNet Captions, YouCook2 and EPIC-KITCHENS-100 can be found [here](https://www.dropbox.com/sh/kn9lp7icfzax48d/AADJFDy5l7LqdRzobtv1cXmKa?dl=0).
Inside the zip file there are three folders: `ActNet`, `Epic` and `YC2`. In each folder there are two pickle files that contains the train and test split: `training_video_features.pkl` and `validation_video_features.pkl`.

## Quick Start Guide
### Requirements
All the dependencies can be found in `ConTra_env.yml`. This project was tested with python 3.8 and pytorch 1.8.0.
### Data
The `data` folder includes the following folders:
* `dataframes`: where you can find the dataframes of ActivityNet Captions, YouCook2 and EPIC-KITCHENS-100.
* `features`: where you can find the features of ActivityNet Captions, YouCook2 and EPIC-KITCHENS-100.
* `models`: where you can find the weights of the pretrained models (ActivityNet weights coming soon...) and the weights of any new run.
* `relevancy`: where you can find the relevancy matrix of EPIC-KITCHENS-100.
* `resources`: where you can find any additional resources to train/test ConTra (i.e. word vocabularies of ActivityNet Captions, YouCook2 and EPIC-KITCHENS-100 as well as the model weights of the S3D model pre-trained on HowTo100M from https://github.com/antoine77340/S3D_HowTo100M).

The `data` folder can be downloaded [here](https://www.dropbox.com/sh/s5mc08xzjo0rxk6/AABofOeByCnFL9w3CLmC6DLFa?dl=0).
### Training

To check all the settings you can change, you can run the following command for YooCook2, ActivityNet Captions and EPIC-KITCHENS-100:
```
python training.py -h
```

If you want to train the model using the best setting when only clip context on Youcook2 is used, run the following command (default values):
```
python training.py --m-video 3 --m-text 0 --lambda4 1  --lambda3 0  --nlayer-video 1 --nhead-video 2 --nlayer-text 1 --nhead-text 2 --embed-dim 512 --dataset YC2

```
The above code will run ConTra using only a sequence of 3 clips as input. If you want to 
python training.py --length-context-video 1 --length-context-text 5 --lambda4 0  --lambda3 1 #m=2
python training.py --length-context-video 1 --length-context-text 7 --lambda4 0  --lambda3 1 #m=3
python training.py --length-context-video 1 --length-context-text 9 --lambda4 0  --lambda3 1 #m=4
python training.py --length-context-video 1 --length-context-text 11 --lambda4 0  --lambda3 1 #m=5


### Testing


## Citation
```
@inproceedings{fragomeni2022ACCV,
  author       = {Fragomeni, Adriano and Wray, Michael and Damen, Dima}
  title        = {ConTra: (Con)text (Tra)nsformer for Cross-Modal Video Retrieval},
  booktitle    = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year         = {2022}
}
```
