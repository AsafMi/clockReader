# This file describes how I created the database
import wandb
import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import os
import cv2 as cv
import math
import pickle
import sys
from tqdm import trange
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch import nn
# from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
# import Functions as fn
# from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from typing import Tuple, Dict, List

# Workflow & steps legend:
# Step 1 : Creation of the dataset folder structure.
#                          V
# Manual work of "undressing" the clock into 3 images (using GIMP):
# "Naked_Clock" , "Hour_Hand" , "Minute_Hand" + extract the center of the clock.
#                          V
# Step 2 : Extracting the time origin from the raw images.
# Step 3 : Use all the above for the creation of the Labeled_Data. (WARNING: !!!Expensive!!!)

steps_activation = [4]

# Choose a folder in which I'll locate all the training data
main_path = r"Clock_Dataset"

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Step 1: Create the folder structure as follows: ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ~~~~~~~ 2 sub-folders: "Raw_Data" and "Labeled_Data"
# ~~~~~~~ Inside "Raw_Data": 50 folders for 50 clocks images ("clock1" ... "clock50")
# ~~~~~~~ Inside "Labels_Data: 720 folders for 720 labels (60 minutes X 12 Hours)
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
rawDirectory = os.path.join(main_path, "Raw_Data")
labeledDirectory = os.path.join(main_path, "Labeled_Data")
if 1 in steps_activation:  # Enter only when the step is activated by the user
    for clockNo in range(50):  # Raw_Data structure creation
        try:
            rawClockPath = os.path.join(rawDirectory, f"clock{clockNo + 1}")
            os.makedirs(rawClockPath)
            print(f"Directory {rawClockPath} created")
        except OSError:
            print(f"Directory {rawClockPath} is already exist")

    for clockNo in range(720):  # Raw_Data structure creation
        try:
            labeledClockPath = os.path.join(labeledDirectory, f"{clockNo}")
            os.makedirs(labeledClockPath)
            print(f"Directory {labeledClockPath} created")
        except OSError:
            print(f"Directory {labeledClockPath} is already exist")


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ V ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ V ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Manual work  - undressing the images ... :(
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ V ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ V ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Step 2: Extracting the time origin from the raw images~ ~ ~ ~ ~ ~ ~ ~ ~
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

if 2 in steps_activation:  # Enter only when this step is activated by the user
    clockNo = 19
    rawClockPath = os.path.join(rawDirectory, f"clock{clockNo}")

    images = {'Hr': cv.imread(os.path.join(rawClockPath, 'Hr.png'), -1),
              'Min': cv.imread(os.path.join(rawClockPath, 'Min.png'), -1)}

    cv.namedWindow('control', 1)
    cv.createTrackbar('Angle_Hr', 'control', 0, 360, lambda x: x)
    cv.createTrackbar('Angle_Min', 'control', 0, 360, lambda x: x)
    Angle_Hr, Angle_Min = 0, 0
    center = tuple(np.genfromtxt(os.path.join(rawClockPath, "center.txt")))
    while ((cv.waitKey(1) & 0xFF) != 27):
        Angle_Hr = cv.getTrackbarPos('Angle_Hr', 'control')
        Angle_Min = cv.getTrackbarPos('Angle_Min', 'control')
        Angle = {'Naked_Clock': 0, 'Hr': Angle_Hr, 'Min': Angle_Min}
        targetImage = cv.imread(os.path.join(rawClockPath, 'Naked Clock.png'), -1)
        for i in images:
            rot_mat = cv.getRotationMatrix2D(center, Angle[i], 1.0)
            image = cv.warpAffine(images[i], rot_mat, images[i].shape[1::-1], flags=cv.INTER_LINEAR)
            alpha_s = image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                targetImage[:, :, c] = (alpha_s * image[:, :, c] +
                                        alpha_l * targetImage[:, :, c])

        cv.imshow('image', targetImage)
    cv.destroyAllWindows()
    time_origin = {'Hr': Angle_Hr, 'Min': Angle_Min}
    with open('time origin.pkl', 'wb') as f:
        pickle.dump(time_origin, f, pickle.HIGHEST_PROTOCOL)

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Step 3: Create the Labeled_Data ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

if 3 in steps_activation or 'All' in steps_activation:  # Enter only when the step is activated by the user
    startTime = time.time()
    for clockNo in trange(50, file=sys.stdout, desc='Clock Number'):
        rawClockPath = os.path.join(rawDirectory, f"clock{clockNo + 1}")

        images = {'Hr': cv.imread(os.path.join(rawClockPath, 'Hr.png'), -1),
                  'Min': cv.imread(os.path.join(rawClockPath, 'Min.png'), -1)}

        center = tuple(np.genfromtxt(os.path.join(rawClockPath, "center.txt")))
        with open(os.path.join(rawClockPath, 'time origin.pkl'), 'rb') as f:
            time_origin = pickle.load(f)
        nakedClockImage = cv.imread(os.path.join(rawClockPath, 'Naked Clock.png'), -1)
        for Hour in trange(12, file=sys.stdout, desc='Hour'):
            for Minute in range(60):
                targetImage = np.copy(nakedClockImage)
                labeledClockPath = os.path.join(labeledDirectory, f"{Hour * 60 + Minute}")
                Angle_Min = time_origin['Min'] - (360 / 60) * Minute
                Angle_Hr = time_origin['Hr'] - (360 / 12) * Hour - (360 / 12) * (Minute / 60)
                Angle = {'Hr': Angle_Hr, 'Min': Angle_Min}
                for i in images:
                    rot_mat = cv.getRotationMatrix2D(center, Angle[i], 1.0)
                    image = cv.warpAffine(images[i], rot_mat, images[i].shape[1::-1], flags=cv.INTER_LINEAR)
                    alpha_s = image[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    for c in range(0, 3):
                        targetImage[:, :, c] = (alpha_s * image[:, :, c] +
                                                alpha_l * targetImage[:, :, c])

                cv.imwrite(os.path.join(labeledClockPath, f"clock{clockNo}.jpg"), targetImage)

        elapsedTime = (time.time() - startTime)
        print("Elapsed time: {0:.1f} Minutes".format(elapsedTime / 60))

#  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Step 4: Model definition ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
#  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
if 4 in steps_activation:

    # wandb might cause an error without this.
    os.environ["WANDB_START_METHOD"] = "thread"

    # Configuration options
    DEVICE = torch.device("cpu")
    IMAGE_SIZE = 64
    BATCHSIZE = 128
    CLASSES = 10
    DIR = os.getcwd()
    LOG_INTERVAL = 10
    N_TRAIN_EXAMPLES = BATCHSIZE * 30
    N_VALID_EXAMPLES = BATCHSIZE * 10
    STUDY_NAME = "pytorch-optimization"
    K_FOLDS = 5
    EPOCHS = 5
    loss_function = nn.MSELoss()

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)

    def class_transform(cls):
        cls = torch.tensor(cls) * 2 * math.pi / 720
        # cls must be torch.tensor and normalized between 0 to 2pi
        return torch.tensor([torch.sin(cls), torch.cos(cls)])

    def reset_weights(m):
        '''
          Try resetting model weights to avoid
          weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    def new_find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {c: int(c) for c in classes}
        return classes, class_to_idx


    def train(optimizer, model, train_loader):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()


    def validate(model, valid_loader):
        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        return accuracy


    def define_model(trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []

        in_features = 28 * 28
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, CLASSES))
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    datasets.ImageFolder.find_classes = new_find_classes
    dataset = datasets.ImageFolder(root=labeledDirectory, transform=data_transform, target_transform=class_transform)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCHSIZE, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCHSIZE, sampler=test_subsampler)

        '''# Sanity check
        def imshow(inp, title=None):
            """Imshow for Tensor."""
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.pause(0.001)  # pause a bit so that plots are updated

        # Get a batch of training data
        inputs, classes = next(iter(trainloader))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=classes)'''


        class SimpleConvNet(nn.Module):
            '''
              Simple Convolutional Neural Network
            '''

            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(3, 10, kernel_size=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear((IMAGE_SIZE-2) * (IMAGE_SIZE-2) * 10, 50),
                    nn.ReLU(),
                    nn.Linear(50, 20),
                    nn.ReLU(),
                    nn.Linear(20, 2)
                )

            def forward(self, x):
                '''Forward pass'''
                return self.layers(x)

        # Init the neural network
        network = SimpleConvNet()
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

        # Run the training loop for defined number of epochs
        for epoch in range(0, EPOCHS):

            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0
            dist_mse = 0.0
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get inputs
                inputs, targets = data

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                with torch.no_grad():
                    # Print statistics
                    current_loss += loss.item()
                    # Set total and correct
                    pred = torch.atan2(outputs.data[:, 0], outputs.data[:, 1]).numpy()*720/(2*np.pi)
                    tar_new = torch.atan2(targets.data[:, 0], targets.data[:, 1]).numpy()*720/(2*np.pi)
                    dist = abs(pred-tar_new)
                    dist_mse += np.mean(np.min(np.concatenate((dist[:,None],(720-dist)[:,None]),axis=1),axis=1))
                    if i % 250 == 249:
                        print('Loss after mini-batch %5d: %.3f' %
                              (i + 1, current_loss / 250))
                        current_loss = 0.0

                        print(f'MSE is {dist_mse/250}')
                        dist_mse = 0.0
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(network.state_dict(), save_path)

        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct
                pred = torch.atan2(outputs.data[:, 0], outputs.data[:, 1]).numpy() * 720 / (2 * np.pi)
                tar_new = torch.atan2(targets.data[:, 0], targets.data[:, 1]).numpy() * 720 / (2 * np.pi)
                total += 1
                correct += np.mean(np.min(np.concatenate((dist[:,None],(720-dist)[:,None]),axis=1),axis=1))


            # Print accuracy
            print('MSE mean for fold %d: %d %%' % (fold, correct / total))
            print('--------------------------------')
            results[fold] = (correct / total)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {K_FOLDS} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')
