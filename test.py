import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from torchsummary import summary
import glob

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

image_transforms = {
	'test' : transforms.Compose([
		transforms.RandomResizedCrop(size=256),
		transforms.CenterCrop(size=224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],
							[0.229, 0.224, 0.225])
	]),
}

dataset = './'
test_directory = os.path.join(dataset, 'test')
print(test_directory)

#Batch Size
bs = 32

data = {
	'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

test_data_size = len(data['test'])
test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)

print(test_data_size)
loss_func = nn.NLLLoss()

test_model = torch.load('_model_9.pt')

def predict(model, test_image_name):


	transform = image_transforms['test']

	test_image = Image.open(test_image_name)
	plt.imshow(test_image)

	test_image_tensor = transform(test_image)

	test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

	with torch.no_grad():
		model.eval()
		# Model outputs log probabilities
		out = model(test_image_tensor)
		ps = torch.exp(out)
		topk, topclass = ps.topk(3, dim=1)
		con ={1:3,2:2,0:1}
		pred = con[topclass.cpu().numpy()[0][0]]
	return pred


data_dir = './test/ts/'
fileOutput = 'result.csv'
fileNme  = []
filePath = []
fullPreds = []

for file in os.listdir(data_dir):
    filePath.append(file)
    fileNme.append(file[:-4])
    fullPreds.append(predict(test_model, data_dir +file))


with open('./' + fileOutput, 'w') as outFile:
	
	for name, prds in zip(fileNme, fullPreds):
		outFile.write('{},{}\n'.format(name,prds))

