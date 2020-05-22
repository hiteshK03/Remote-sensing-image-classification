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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class myData(Dataset):

    train_file = "training.csv"
    test_file = "testing.csv"

    def __init__(self, data_dir, transform=None, train=True):
        
        # The transform is goint to be used on image
        self.transform = transform
        
        if train:
            self.data_dir = os.path.join(data_dir,"train")
            data_file = self.train_file
        else:
            self.data_dir = os.path.join(data_dir,"test")
            data_file = self.test_file
        
        data_dircsv_file=os.path.join(self.data_dir,data_file)
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        # Image file path
        img_name = os.path.join(self.data_dir,self.data_name.iloc[idx, 0])
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 1]

        target = {}
        target = {'aircrafts':0, 'ships':1, 'none':2}

        y = target[y]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y

class splitData(Dataset):

    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y
        
    def __len__(self):
        return len(self.subset)

def makeTransform():
    transform_train = transforms.Compose([transforms.RandomResizedCrop(size=256, scale = (0.8,1.0)),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

    transform_test = transforms.Compose([transforms.Resize(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

    transform = {}
    transform = {'train': transform_train, 'test': transform_test}

    return transform

def trainValid(model, src_dir, val_set_fraction, batch_size=32, epochs=25):
    '''
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''

    transform = makeTransform()

    train_valid = myData(data_dir=src_dir, transform=transform['train'], train=True)

    lengths = [int(len(train_valid)*(1-val_set_fraction)), int(len(train_valid)*val_set_fraction)]
    train, val = random_split(train_valid, lengths)

    train_set = splitData(train)
    val_set = splitData(val)

    train_length = len(train_set)
    val_length = len(val_set)

    trainloader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle=False, drop_last=True)
    valloader = DataLoader(dataset = val_set, batch_size = 4, shuffle=False, drop_last=True)

    lossDeterminer = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    expLrScheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    start = time.time()
    history = []
    bestAcc = 0.0
    bestEpoch = 0

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
        
        for i, (inputs, labels) in enumerate(trainloader):

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
            for j, (inputs, labels) in enumerate(valloader):
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
        trainLossAvg = trainLoss/train_length
        trainAccAvg = trainAcc/train_length

        # Find average training loss and training accuracy
        validLossAvg = validLoss/val_length
        validAccAvg = validAcc/val_length

        history.append([trainLossAvg, validLossAvg, trainAccAvg, validAccAvg])
                
        epochEnd = time.time()
    
        print("Epoch : {:03d}, Training: Loss : {:.4f}, Accuracy: {:.4f}%".format(epoch, trainLossAvg, trainAccAvg*100))
        print("Validation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(validLossAvg, validAccAvg*100, epochEnd-epochStart))
        
        # Save if the model has best accuracy till now
        if validAccAvg > bestAcc:
            bestAcc = validAccAvg
            bestEpoch = epoch
            torch.save(model, src_dir+'/model'+'.pt')
            print("model for epoch {} saved".format(bestEpoch))

            
        print("Best accuracy achieved so far : {:.4f} on epoch {}".format(bestAcc, bestEpoch))
    
    # Plot and save train and validation losses
    history = np.array(history)
    plt.plot(history[:,0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0,1)
    plt.savefig(directory+'lossCurve.png')
    plt.show()

    # Plot and save train and validation accuracies
    plt.plot(history[:,2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.savefig(directory+'accuracyCurve.png')
    plt.show()

def computeTestSetAccuracy(model, src_dir):

    # _, _, testloader, _, _, test_length  = getData()

    transform = makeTransform()

    test_set = myData(data_dir=src_dir, transform=transform['test'], train=False)
    test_length = len(test_set)

    testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=False)

    loss_criterion = nn.CrossEntropyLoss()

    test_acc = 0.0
    test_loss = 0.0

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_length
    avg_test_acc = test_acc/test_length

    print("Test accuracy : " + str(avg_test_acc))