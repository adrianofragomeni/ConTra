# ConTra
This repo contains code and data of ConTra model used in ConTra: (Con)text (Tra)nsformer for Cross-Modal Video Retrieval [1].

## Features
The video features of ActivityNet Captions, YouCook2 and EPIC-KITCHENS-100 can be found [here](https://www.dropbox.com/s/b3mhjdn3tcb3xhc/features.zip?dl=0).
Inside the zip file there are three folders: `ActNet`, `Epic` and `YC2`. In each folder there are two pickle files that contains the train and test split: `training_video_features.pkl` and `validation_video_features.pkl`.

## Quick Start Guide
### Requirements
All the dependencies can be found in `ConTra_env.yml`. This project was tested with python 3.8 and pytorch 1.8.0.
### Data
The `data` folder includes the following folders:
* `dataframes`: where you can find the dataframes of ActivityNet Captions, YouCook2 and EPIC-KITCHENS-100.
* `features`: where you can find the features of ActivityNet Captions, YouCook2 and EPIC-KITCHENS-100.
* `models`: where you can find the weights of the pretrained models and the weights of any new run.
* `relevancy`: where you can find the relevancy matrix of EPIC-KITCHENS-100.
* `resources`: where you can find any additional resources to train/test ConTra (i.e. word vocabularies of ActivityNet Captions, YouCook2 and EPIC-KITCHENS-100).
The `data` folder can be downloaded [here](prova). (Remember to upload all the weights in the zip file!!!!!!!!!)
### Training
Code coming soon...
### Testing
Code coming soon...
### Citation

# References
[1]
