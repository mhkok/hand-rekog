# USAGE
# python train.py
# import the necessary packages
from datasets import load_dataset
from config import config
#from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
#from sklearn.model_selection import train_test_split
#from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
#import cv2
import os


# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

train_loader = load_dataset("Francesco/hand-gestures-jps7z", split="train", batch_size=30)

for key, value in train_loader(batch_size = 30):
    print(key, value)
    #image = row["image"]



    #data.append(image)    