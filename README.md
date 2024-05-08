## RSL Pretext ## 
This is the code for the RSL pretext, that is used for self-supervised visual representation learning from videos.
The trained models for the Ablation experiments are available at [RSL-Ablation](https://drive.google.com/drive/folders/1_7uOuHol2psBiaaSOuOKKFtReTUg-BCk?usp=drive_link).
The trained models for the Main experiments are available at [RSL-Models](https://drive.google.com/drive/folders/1-YgWMN-sqwYQuWI7FD1NZKA04Ep9xOus?usp=drive_link).

### Notes: ###
1. The dataset (UCF-101 and HMDB-51) main folder should have two sub-folders named (Videos, Splits). The Videos sub-folder contains all the videos of the dataset, while the Splits sub-folder contains the files that define the videos for each split.
2. Each experiment should have a name that is configured using the parse_args function. In addition, there should be a folder with the same experiment's name
inside the experiments folder. Each experiment folder should have a sub-folder named Run. 

We used a workstaion that has a conda environment with the following packages:
+ tqdm 4.64.1
+ pandas 1.5.2
+ python 3.8.15
+ pytorch 1.13.0
+ torchaudio 0.13.0
+ tensorboardx 2.5.1
+ torchsummary 1.5.1
+ torchvision 0.14.0
+ scikit-learn 1.1.3
+ scikit-video 1.1.11
+ ffmpeg-python 0.2.0
+ opencv-python 4.6.0.66


