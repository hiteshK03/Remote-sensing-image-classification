import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import time
import argparse
from utils import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", default = './../src',
	help = "Path to the dirctory where images are stored")
ap.add_argument("-b", "--batch", type=int, default=32,
	help="training batch size")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="no. of epochs")
ap.add_argument("-v", "--val", type=float, default=0.1,
	help="Fraction of data to be used as validation set")
ap.add_argument("-t", "--test", type=bool, default=False,
	help="To test the model")
args = vars(ap.parse_args())

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

directory = args["directory"]

num_classes = 3

if args["test"] == False:
	print("[INFO] loading model...")
	vgg19 = models.vgg19_bn(pretrained=True)

	# Freeze model parameters
	for param in vgg19.parameters():
	    param.requires_grad = False

	vgg19.classifier = nn.Sequential(nn.Linear(25088, 4096),
	nn.ReLU(),
	nn.Dropout(0.4),
	nn.Linear(4096, 1024),
	nn.ReLU(),
	nn.Dropout(0.4),
	nn.Linear(1024, num_classes),
	nn.LogSoftmax(dim=1))

	vgg19.to(device)

	# Print the model to be trained
	summary(vgg19, input_size=(3, 224, 224), batch_size=args["batch"])

	# Train the model
	trainValid(vgg19, src_dir=directory, val_set_fraction=args["val"], batch_size=args["batch"], epochs=args["epochs"])

else:
	# Load the model with best validation accuracy
	saved_model = torch.load(directory+'/model'+'.pt')

	# Test the loaded model
	computeTestSetAccuracy(saved_model, src_dir=directory)
