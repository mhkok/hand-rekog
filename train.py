## Largely based on https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/

from datasets import load_dataset
from config import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from dataset import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from PIL import Image

wandb.require("core")
wandb.login()
run = wandb.init(project="hand-rekog")

# W&B: Log hyperparameters using config
cfg = wandb.config
cfg.update(
    {
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "lr": config.INIT_LR,
    }
)

# loading and preprocessing the hand gesture data
train_loader = load_dataset("Francesco/hand-gestures-jps7z", split="train")
test_loader = load_dataset("Francesco/hand-gestures-jps7z", split="test")

train_data, train_targets = preprocessing.prepare_data(train_loader)
test_data, test_targets = preprocessing.prepare_data(test_loader)

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
)

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the modelx
# summary
opt = Adam(learning_rate=1e-4)
model.compile(loss="mse", optimizer=opt)
callbacks = [EarlyStopping(patience=10), WandbMetricsLogger()]
print(model.summary())

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
    x=train_data,
    y=train_targets,
    validation_data=(test_data, test_targets),
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=1,
    callbacks=callbacks
)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH)

# Visualize the model
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)