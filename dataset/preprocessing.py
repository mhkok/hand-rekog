from datasets import load_dataset
import io
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np


def coco_to_pascal_voc(coco_bbox):
    x, y, width, height = coco_bbox
    x_min = x
    y_min = y
    x_max = x + width
    y_max = y + height
    return [x_min, y_min, x_max, y_max]


def prepare_data(dataset):
    train_data = []
    test_data = []
    eval_data = []
    train_targets = []
    test_targets = []

    for hand in dataset:
        img = hand["image"]

        (h, w) = hand["width"], hand["height"]
        bbox = hand["objects"].get("bbox")

        if bbox and len(bbox) > 0:
            coco_bbox = bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]

            x_min, y_min, x_max, y_max = coco_to_pascal_voc(coco_bbox)

            x_min = float(x_min) / w
            y_min = float(y_min) / h
            x_max = float(x_max) / w
            y_max = float(y_max) / h

            image_bytes = io.BytesIO()
            img.save(image_bytes, format="JPEG")
            image_bytes.seek(0)

            image = load_img(image_bytes, target_size=(224, 224))
            image = img_to_array(image)

        else:
            print("Bounding box not found for this hand.")
            continue

        if dataset.split == "train":
            train_data.append(image)
            train_targets.append([x_min, y_min, x_max, y_max])

        if dataset.split == "test":
            test_data.append(image)
            test_targets.append([x_min, y_min, x_max, y_max])

        else:
            eval_data.append(image)

    if dataset.split == "train":
        # convert to numpy array
        train_data = np.array(train_data, dtype="float32") / 255.0
        train_targets = np.array(train_targets, dtype="float32")

        return train_data, train_targets
    if dataset.split == "test":
        # convert to numpy array
        test_data = np.array(test_data, dtype="float32") / 255.0
        test_targets = np.array(test_targets, dtype="float32")
        return test_data, test_targets
    else:
        eval_data = np.array(eval_data, dtype="float32") / 255.0
        return eval_data
        
