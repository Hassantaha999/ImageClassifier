# Basic usage: python predict.py /path/to/image checkpoint
# predict the category of a particular image 

import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json


def load_model_checkpoint(args):
    # load checkpoint of the model 
    checkpoint = torch.load(args.model_chk_path)

    ## decide which pre-trained model to use based on the user's input  
    # ResNet models    
    if checkpoint['model'] == "resnet18":
        model = models.resnet18(pretrained=True)
    elif checkpoint['model'] == "resnet34":
        model = models.resnet34(pretrained=True)
    elif checkpoint['model'] == "resnet50":
        model = models.resnet50(pretrained=True)
    # VGG models 
    elif checkpoint['model'] == "vgg13":
        model = models.vgg13(pretrained=True)
    elif checkpoint['model'] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif checkpoint['model'] == "vgg19":
        model = models.vgg19(pretrained=True)
    # DenseNet models
    elif checkpoint['model'] == "densenet121":
        model = models.densenet121(pretrained=True)
    elif checkpoint['model'] == "densenet169":
        model = models.densenet169(pretrained=True)
    elif checkpoint['model'] == "densenet161":
        model = models.densenet161(pretrained=True)

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    #model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
    
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Define the transformations for resizing, cropping, and normalizing
    preprocess = transforms.Compose([transforms.Resize(224),
                                     transforms.RandomRotation(45),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    
    # Open the image using PIL
    img = Image.open(image_path)
    
    # Apply the defined transformations
    img = preprocess(img)
    
    # Convert the image to a NumPy array
    #img = np.array(img)
    
    return img

def predict(args, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Set the model to evaluation mode and move it to the GPU CUDA
    if torch.cuda.is_available() and args.gpu == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # move model to the choosen device 
    model.to(device)
    model.eval()

    # Preprocess the image
    img = process_image(args.image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    # Perform the inference
    with torch.no_grad():
        logps = model.forward(img.to(device))

    # Calculate class probabilities and get the topk predictions
    print("###########################")
    print("Calculating Probabilities...")
    print("###########################")
    probabilities = torch.exp(logps)
    top_probabilities, top_indices = probabilities.topk(args.top_k)

    # Convert the tensors to lists
    top_probabilities = top_probabilities.tolist()[0]
    top_indices = top_indices.tolist()[0]

    # Convert the indices to class labels (if available)
    #model.class_to_idx = image_datasets_train['train'].class_to_idx
    #idx_to_class = {int(k): v for k, v in model.class_to_idx.items()}
    #top_classes = [idx_to_class[idx] for idx in top_indices]
    
    with open(args.category_names_path, 'r') as f:
        cat_to_name = json.load(f)

    cat_to_name_int = {int(k): v for k, v in cat_to_name.items()}
    top_classes_dict = {idx: cat_to_name_int[idx] for idx in top_indices}
    top_classes = [cat_to_name_int[idx] for idx in top_indices]

    print("###########################")
    print("The Results of the Predictions:")
    print("###########################")
    for i in range(args.top_k):
        result = f"{i}) Category: {top_classes[i]} (idx: {list(top_classes_dict.keys())[i]}) has a probability of {top_probabilities[i]*100:.3f}."
        # save the result into a .txt file. 
        with open("predictions_topk_results.txt", "a") as output_file:
            output_file.write(result + "\n")
        print(result)
    
    return top_probabilities, top_classes_dict, top_classes

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def display_image_topk(args, top_probabilities, top_classes):
    
    image = process_image(args.image_path)
    
    ## Plots    
    # input image of the flower 
    imshow(image, ax=plt)
    
    # probabilities for the topk classes as a bar graph
    y_labels = [i for i in np.arange(args.top_k)]
    
    plt.figure()
    plt.barh(y_labels, top_probabilities)
    plt.yticks(y_labels, top_classes)
    plt.xlabel("Probability")
    plt.ylabel("Flower Name")
    #plt.show()
    #plt.savefig("predicitions_topk_results.png")
    

if __name__ == '__main__':
    
    ## define input parameters:
    parser = argparse.ArgumentParser()
    # required inputs
    parser.add_argument('image_path', 
                        help="The path for the image to classify.")
    parser.add_argument('model_chk_path', 
                        help="The path for checkpoint for the model to be used.")
    # optional arguments
    parser.add_argument('--category_names_path', default='cat_to_name.json',
                        help="The path for json file that maps categories to the real name of the different catogeries (default --> cat_to_name.json).")
    parser.add_argument('--top_k', action='store', default=5, type=int,
                        help="The number of the categories with their probabilities to be printed (default --> 5).")
    parser.add_argument('--gpu', action='store', default="gpu",
                        help="Include if you want to use the GPU in training your model (default --> gpu).")
    parser.add_argument('--plot', action='store', default=False, type=bool,
                        help="Decide if one wants to plot the results of the predicitons (default --> False)")
    
    # Parse and print the results
    args = parser.parse_args()

    # load the model checkpoint 
    model = load_model_checkpoint(args)

    # make predictions
    top_probabilities, top_classes_dict, top_classes = predict(args, model)

    # plot the results 
    if args.plot == True: 
        display_image_topk(args, top_probabilities, top_classes)