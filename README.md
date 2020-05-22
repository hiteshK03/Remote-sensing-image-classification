## Overview
Transfer learning applied to train an image classifier for classifying remote sensing data into three classes:
* aircrafts
* ships
* none

# Remote Sensing Image Classification
### Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Running](#running)

## Installation
The program requires the following dependencies (easy to install using pip: pip3 install -r requirements.txt):
* python 3.5
* pytorch
* numpy
* pandas
* matplotlib
* Pillow
* CUDA (for using GPU)

## Dataset

The dataset can be downloaded from [here](https://drive.google.com/file/d/1BEmmtKygwxH6U3rNpUJrgH6YRhJIfmxC/view?usp=sharing)
After downloading it can be extracted by:
```
unzip src.zip
```
The structure of extracted folder is shown below:
```bash
src
├── test
│   ├── 1951.png
│   ├── 1977.png
│   ├── 1999.png
│   └── testing.csv
├── train
│   ├── aircrafts
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── 3.png
│   ├── none
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── 3.png
│   ├── ships
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── 3.png
│   └── training.csv
├── main.py
└── utils.py
```
## Running

To train the model, simply run ```python3 main.py```. Once trained, you can test the results with ```python3 main.py --test True``` (make sure that you have a saved model file : ```model.pt``` before testing)

Here are some flags which could be useful. For more help and options, use ```python3 main.py -h```:

- --directory <name> : if the current directory is not ```src```.
- --batch <number> : to change the training batch size (default = 32)
- --epochs <number> : to change the number of epochs (default = 25)
- --val <fraction> : to change the fraction of validation set out of total training set (default = 0.1)
