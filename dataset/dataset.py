from datasets import load_dataset

# from config import config
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import wandb
from PIL import Image
import io

wandb.login()

WANDB_PROJECT = "hand-rekog"

dataset = load_dataset("Francesco/hand-gestures-jps7z", data_dir="data")

train_loader = load_dataset("Francesco/hand-gestures-jps7z", split="train")
test_loader = load_dataset("Francesco/hand-gestures-jps7z", split="test")

train_data = []
test_data = []
train_targets = []
test_targets = []
filenames = []

def coco_to_pascal(coco_bbox):
    x_min, y_min, width, height = coco_bbox
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]


def prepare_data(dataset):
    for hand in dataset:
        img = hand["image"]

        (h, w) = hand["width"], hand["height"]
        bbox = hand["objects"].get("bbox")

        if bbox and len(bbox) > 0:
            x_min, y_min, width, height = bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]

            x_min = float(x_min) / w
            y_min = float(y_min) / h
            width = float(width) / w
            height = float(height) / h

            image_bytes = io.BytesIO()
            img.save(image_bytes, format="JPEG")
            image_bytes.seek(0)

            image = load_img(image_bytes, target_size=(224, 224))
            # print(image.size)
            image = img_to_array(image)

        else:
            print("Bounding box not found for this hand.")
            continue

        if dataset == train_loader:
            train_data.append(image)
            train_targets.append([x_min, y_min, width, height])
        else:
            dataset == test_loader
            test_data.append(image)
            test_targets.append([x_min, y_min, width, height])

    return train_data, test_data, test_targets, train_targets


prepare_data(train_loader)
prepare_data(test_loader)

test_data = np.array(test_data, dtype="float32") / 255.0
test_targets = np.array(test_targets, dtype="float32") 
train_data = np.array(train_data, dtype="float32") / 255.0
train_targets = np.array(train_targets, dtype="float32")

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
opt = Adam(lr=1e-4)
model.compile(loss="mse", optimizer=opt)
print(model.summary())
print(train_data.shape)
print(train_targets.shape)

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
    x=train_data, y=train_targets,
    validation_data=(test_data, test_targets),
    batch_size=32,
    epochs=20,
    verbose=1,
)
