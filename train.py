import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch, torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import time
from torchsummary import summary

image_transforms = {
	'train' : transforms.Compose([
		transforms.RandomResizedCrop(size=256, scale = (0.8,1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
		transforms.CenterCrop(size=224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],
							[0.229, 0.224, 0.225])
	]),
	'valid' : transforms.Compose([
		transforms.RandomResizedCrop(size=256),
		transforms.CenterCrop(size=224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],
							[0.229, 0.224, 0.225])
	]),
}

dataSet = './'

trainDir = os.path.join(dataSet, 'train')
validationDir = os.path.join(dataSet, 'valid')

#Batch Size
bs = 32

#Number of classes
numClasses = len(os.listdir(validationDir))-1
print(numClasses)

#Data load
data = {
	'train': datasets.ImageFolder(root=trainDir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=validationDir, transform=image_transforms['valid']),
}

# Index mapping
idxClass = {v: k for k, v in data['train'].classIdx.items()}
print(idxClass)

trainDataSize = len(data['train'])
validationDataSize = len(data['valid'])

trainDataLoader = DataLoader(data['train'], batch_size=bs, shuffle=True)
validationDataLoader = DataLoader(data['valid'], batch_size=bs, shuffle=True)

# Print the train, validation set data sizes
trainDataSize, validationDataSize

vgg19 = models.vgg19_bn(pretrained=True)
# Freeze model parameters
# Same for all
for param in vgg19.parameters():
    param.requires_grad = False

vgg19.classifier = nn.Sequential(nn.Linear(25088, 4096),
nn.ReLU(),
nn.Dropout(0.4),
nn.Linear(4096, 1024),
nn.ReLU(),
nn.Dropout(0.4),
nn.Linear(1024, numClasses +1),
nn.LogSoftmax(dim=1))

# Define Optimizer and Loss Function
lossFunc = nn.NLLLoss()
optimizer = optim.Adam(vgg19.parameters(), lr=1e-2)

# Decay LR by a factor of 0.1 every 7 epochs
expLrScheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

def trainValid(model, lossDeterminer, optimizer, epochs=25):
    '''
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    device = torch.device("cpu")
    start = time.time()
    history = []
    bestAcc = 0.0

    for epoch in range(epochs):
        epochStart = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        trainLoss = 0.0
        trainAcc = 0.0
        
        validLoss = 0.0
        validAcc = 0.0
        
        for i, (inputs, labels) in enumerate(trainDataLoader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = lossDeterminer(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to trainLoss
            trainLoss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            _, predictions = torch.max(outputs.data, 1)
            corrCounts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert corrCounts to float and then compute the mean
            acc = torch.mean(corrCounts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to trainAcc
            trainAcc += acc.item() * inputs.size(0)
            
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(validationDataLoader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = lossDeterminer(outputs, labels)

                # Compute the total loss for the batch and add it to validLoss
                validLoss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                corrCounts = predictions.eq(labels.data.view_as(predictions))

                # Convert corrCounts to float and then compute the mean
                acc = torch.mean(corrCounts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to validAcc
                validAcc += acc.item() * inputs.size(0)

                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        trainLossAvg = trainLoss/trainDataSize 
        trainAccAvg = trainAcc/trainDataSize

        # Find average training loss and training accuracy
        validLossAvg = validLoss/validationDataSize 
        validAccAvg = validAcc/validationDataSize

        history.append([trainLossAvg, validLossAvg, trainAccAvg, validAccAvg])
                
        epochEnd = time.time()
    
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n")
        print("Validation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, trainLossAvg, trainAccAvg*100, validLossAvg, validAccAvg*100, epochEnd-epochStart))
        
        # Save if the model has best accuracy till now
        modelForTest = torch.save(model, dataSet+'_model_'+str(epoch)+'.pt')
            
    return model, history

# Print the model to be trained
summary(vgg19, input_size=(3, 224, 224), batch_size=bs, device='cpu')

# Train the model for 25 epochs
totalEpochs = 10
trainedModel, history = trainValid(vgg19, lossFunc, optimizer, totalEpochs)

torch.save(history, dataSet+'_history.pt')

history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.savefig(dataSet+'lossCurve.png')
plt.show()

plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig(dataSet+'accuracyCurve.png')
plt.show()






