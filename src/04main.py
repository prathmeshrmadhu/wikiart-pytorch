import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import os, sys
import pickle as pkl
import argparse
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from training import train_model

from dataset import wikiart_dataset

from loss import FocalLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument('--root', '-root', default='.', help='path to root dir, should have images folder at same level')
ap.add_argument('--data_folder', '-data_folder', required=True, help='path to data folder for training')
ap.add_argument('--train_csv_path', '-train_csv_path', required=True, help='path to the training data csv')
ap.add_argument('--valid_csv_path', '-valid_csv_path', required=True, help='path to the validation data csv')
ap.add_argument('--batch_size', '-batch_size', type=int, default=16, help='size of batch for training')
ap.add_argument('--image_size', '-image_size', type=int, default=256, help='image size used for training before 224 crop')
ap.add_argument('--dropout', '-dropout', default=0.1, help='percentage of dropout used to regularize the training')
ap.add_argument('--num_epochs', '-num_epochs', type=int, default=2, help='number of epochs for training')
ap.add_argument('--num_classes', '-num_classes', type=int, default=23, help='number of classes for training, style: 27, artist: 24, genre: 10')
ap.add_argument('--flag', '-flag', required=True, default=23, help='which classification: style, artist, genre')
ap.add_argument('--loss', '-loss', type=str, default='BCE', help='which extra loss for training')

args = vars(ap.parse_args())

ROOT_PATH = args['root']
DATASET_PATH = args['data_folder']
BATCH_SIZE = args['batch_size']
DROPOUT = args['dropout']
NUM_EPOCHS = args['num_epochs']
NUM_CLASSES = args['num_classes']
FLAG = args['flag']
LOSS = args['loss']

if FLAG == 'style':
    NUM_CLASSES = 27
elif FLAG == 'artist':
    NUM_CLASSES = 24
elif FLAG == 'genre':
    NUM_CLASSES = 10
else:
    sys.exit('Please pass the correct flag as "style", "artist" or "genre"')

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((args['image_size'],args['image_size'])),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

## Load Dataset and Create DataLoader
train_csv = args['train_csv_path']
valid_csv = args['valid_csv_path']

train_dataset = wikiart_dataset(data_folder=args['data_folder'], csv=train_csv, num_classes=NUM_CLASSES, transform=data_transforms['train'])
valid_dataset = wikiart_dataset(data_folder=args['data_folder'], csv=valid_csv, num_classes=NUM_CLASSES, transform=data_transforms['valid'])
    
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True)

dataloaders_dict = {'train': train_loader,
                   'valid': valid_loader}

dataset_sizes = {'train':len(train_dataset),
                'valid':len(valid_dataset)}

## DEFINING resnet50 MODEL
model = models.resnet50(pretrained=True).to(device)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_ft = model_ft.to(device)

## Criteria for training
if LOSS == 'focal':
    criterion = FocalLoss()
else:
    criterion = nn.BCEWithLogitsLoss()


## DEFINING OPTIMIZERS
lr = 0.0001
params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

## TRAINING LOOP
model_finetuned, train_acc_history, val_acc_history, train_loss_history, val_loss_history = train_model(
        model_ft, dataloaders_dict, criterion, optimizer, exp_lr_scheduler, nb_classes=NUM_CLASSES, num_epochs=NUM_EPOCHS)

## SAVING THE MODEL
CURR_PATH = os.getcwd()
model_name = FLAG + '_wikiart_classification_' + str(LOSS) + '_loss_epochs_' + str(NUM_EPOCHS) + '.pth'
loss_name = 'losses_' + model_name.split('.')[0] + '.png'

if not os.path.isdir(os.path.join(CURR_PATH, 'resources/models')):
    os.makedirs(os.path.join(CURR_PATH, 'resources/models'))
model_save_path = 'resources/models/'+str(model_name)
torch.save(model_finetuned.state_dict(), model_save_path)

## Saving the train/val accuracies
if not os.path.isdir(os.path.join(CURR_PATH, 'resources/histories')):
    os.makedirs(os.path.join(CURR_PATH, 'resources/histories'))

train_history_save_path = os.path.join(CURR_PATH, 'resources/histories/' + model_name.split('.')[0] + '_train.pkl')
valid_history_save_path = os.path.join(CURR_PATH, 'resources/histories/' + model_name.split('.')[0] + '_valid.pkl')

with open(train_history_save_path, 'wb') as out:
    pkl.dump(train_acc_history, out)
with open(valid_history_save_path, 'wb') as out:
    pkl.dump(val_acc_history, out)

## Saving the train/val losses
if not os.path.isdir(os.path.join(CURR_PATH, 'resources/losses')):
    os.makedirs(os.path.join(CURR_PATH, 'resources/losses'))

train_loss_history_save_path = os.path.join(CURR_PATH, 'resources/losses/losses_' + model_name.split('.')[0] + '_train.pkl')
valid_loss_history_save_path = os.path.join(CURR_PATH, 'resources/losses/losses_' + model_name.split('.')[0] + '_valid.pkl')

with open(train_loss_history_save_path, 'wb') as out:
    pkl.dump(train_loss_history, out)
with open(valid_loss_history_save_path, 'wb') as out:
    pkl.dump(val_loss_history, out)