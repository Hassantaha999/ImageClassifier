# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Advanced Options:
#         1) Set directory to save checkpoints: (python train.py data_dir --save_dir save_directory)
#         2) Choose architecture: (python train.py data_dir --arch "vgg13")
#         3) Set hyperparameters: (python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 --dropout 0.5)
#         4) Use GPU for training: (python train.py data_dir --gpu)
# usage example: python train.py './flowers' './saved_data' --epochs 5

# needed modules and packages
import argparse
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import os 

def transformations(data_path):
    # Define the directories for the train, validation, and test datasets
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'

    # Define your transforms for the training, validation, and testing sets
    data_transforms_train = transforms.Compose([transforms.Resize(224),
                                                transforms.RandomRotation(45),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    data_transforms_valid_test = transforms.Compose([transforms.Resize(224),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    image_datasets_train = ImageFolder(root=train_dir, transform=data_transforms_train)
    image_datasets_valid = ImageFolder(root=valid_dir, transform=data_transforms_valid_test)
    image_datasets_test = ImageFolder(root=test_dir, transform=data_transforms_valid_test)

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train = data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    dataloaders_valid = data.DataLoader(image_datasets_valid, batch_size=64, shuffle=True)
    dataloaders_test = data.DataLoader(image_datasets_test, batch_size=64, shuffle=True)

    return dataloaders_train, dataloaders_valid, dataloaders_test


def training(args, train_data_loader, valid_data_loader):

    ## decide which pre-trained model to use based on the user's input  
    # ResNet models    
    if args.arch == "resnet18":
        model = models.resnet18(pretrained=True)
    elif args.arch == "resnet34":
        model = models.resnet34(pretrained=True)
    elif args.arch == "resnet50":
        model = models.resnet50(pretrained=True)
    # VGG models 
    elif args.arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif args.arch == "vgg19":
        model = models.vgg19(pretrained=True)
    # DenseNet models
    elif args.arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif args.arch == "densenet169":
        model = models.densenet169(pretrained=True)
    elif args.arch == "densenet161":
        model = models.densenet161(pretrained=True)

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # build the classifier
    input_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_features, args.hidden_units, bias=True)),
                            ('relu1', nn.ReLU()),
                            ('dropout', nn.Dropout(args.dropout)),
                            ('fc2', nn.Linear(args.hidden_units, args.output_features, bias=True)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    # define criterion 
    criterion = nn.NLLLoss()   

    # define optimizer 
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # decide which device to use based on the user's input   
    if torch.cuda.is_available() and args.gpu == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # move model to the choosen device 
    model.to(device)

    # train the model
    print("######################")
    print("Start Training...")
    print("######################")

    epochs = args.epochs
    steps = 0
    print_freq = args.print_freq

    for e in range(epochs):
        running_loss = 0
        
        for Images, labels in train_data_loader:
            steps += 1

            Images, labels = Images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(Images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_freq == 0:
                model.eval()
                
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for Images, labels in valid_data_loader:
                        Images, labels = Images.to(device), labels.to(device)

                        log_ps = model.forward(Images)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Step {steps}.. "
                      f"Loss: {running_loss/print_freq:.3f}.. "
                      f"Validation Loss: {valid_loss/len(valid_data_loader):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_data_loader):.3f}")
                
                running_loss = 0
                
                model.train()
    
    print("######################")
    print("Finished Training!")
    print("######################")
    
    # Save model 
    #model.class_to_idx = train_data_loader.class_to_idx
    torch.save({'output_size': args.output_features,
            'model': args.arch,
            'learning_rate': args.learning_rate,
            'classifier': model.classifier,
            'epochs': args.epochs,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict()},
            #'class_to_idx': model.class_to_idx}, 
            args.save_dir)
    
    print("###########################")
    print("#### Checkpoint Saved! ####")
    print("###########################")

if __name__ == "__main__":
    
    ## define input parameters:
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help="Directory of the training images. (default --> ./flowers/)")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth",
                        help="Directory were the model will be saved. (default --> ./checkpoint.pth)")
    parser.add_argument('--arch', action="store", default="vgg16",
                        help="Define the type of the pre-trained model. Available models: [vgg13, vgg16, vgg19, resnet18, resnet34, resnet50, densenet121, densenet169, densenet161] (default --> vgg16).")
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001,
                        help="Set the learning rate (default --> 0.001).")
    parser.add_argument('--hidden_units', action="store", type=int, dest="hidden_units", default=1024,
                        help="Set the number of units in the hidden layer (default --> 2048).")
    parser.add_argument('--epochs', action="store", type=int, default=3,
                        help="Set the number of epochs (default --> 3).")
    parser.add_argument('--dropout', action="store", type=float, default=0.2,
                        help="Set the value for the dropout (default --> 0.2).")
    parser.add_argument('--gpu', action="store", default="gpu",
                        help="Include if you want to use the GPU in training your model (default --> gpu).")
    parser.add_argument('--print_freq', action="store", type=int, default=10,
                        help="Set the frequency with which the steps of training the model should be printed to the user. (default --> 10)")
    parser.add_argument('--output_features', action="store", type=int, default=102,
                        help="Number of output features. (default --> 102)")
    
    # save the inputs into variables 
    args = parser.parse_args()

    # Make the necessary transformation on the datasets
    data_train, data_valid, data_test = transformations(args.data_dir)

    # Train the model 
    training(args, data_valid, data_test)