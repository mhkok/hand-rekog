import wandb
from config import config
import io
from datasets import load_dataset
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import cv2
import numpy as np
from dataset import preprocessing

def bounding_boxes(filename, x_min, y_min, x_max, y_max, class_id, log_width, log_height):
    # load raw input photo
    raw_image = load_img(filename, target_size=(log_height, log_width))
    all_boxes = []
    # plot each bounding box for this image
        # get coordinates and labels
    box_data = {"position" : {
        "minX" : x_min,
        "maxX" : x_max,
        "minY" : y_min,
        "maxY" : y_max},
        "class_id" : class_id,
        # optionally caption each box with its class and score
        #"box_caption" : "%s (%.3f)" % (v_labels[b_i], v_scores[b_i]),
        "domain" : "pixel"}
        #"scores" : { "score" : v_scores[b_i] }}
    all_boxes.append(box_data)


    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    box_image = wandb.Image(raw_image, boxes = {"predictions": {"box_data": all_boxes}})
    return box_image


print("[INFO]LOAD EVAL DATASET")
eval_loader = load_dataset("Francesco/hand-gestures-jps7z", split="validation")
eval_data = preprocessing.prepare_data(eval_loader)

print("[INFO] loading object detector...")
print("[INFO] MODEL PATH: ", config.MODEL_PATH)
model = load_model(config.MODEL_PATH)

wandb.require("core")
wandb.login()
run = wandb.init(project="hand-rekog")

# ✨ W&B: Create a Table to store predictions for each test step
columns = ["img_ground_truth_bbox", "img_pred_bbox"]
# for digit in range(1):
#     columns.append("score_" + str(digit))
test_table = wandb.Table(columns=columns)

for hand in eval_loader:
    image = hand["image"]
    
    #convert to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    
    # get bounding box & convert to pascal voc
    bbox_label = hand["objects"].get("bbox")
    (h, w) = hand["width"], hand["height"]

    if bbox_label and len(bbox_label[0]) >= 4:
        coco_bbox = bbox_label[0][0], bbox_label[0][1], bbox_label[0][2], bbox_label[0][3]
        x_min, y_min, x_max, y_max = preprocessing.coco_to_pascal_voc(coco_bbox)
    else:
        continue

    class_id = int(hand["objects"].get("category", 0)[0])
    
    img_ground_truth_bbox = bounding_boxes(image_bytes, x_min, y_min, x_max, y_max, class_id, w, h)
    
    image = load_img(image_bytes, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)[0]

    x_min_pred, y_min_pred, x_max_pred, y_max_pred = map(float, preds[:4]) #converting to float from numpy
    
    x_min_pred = int(x_min_pred * w)
    y_min_pred = int(y_min_pred * h)
    x_max_pred = int(x_max_pred * w)
    y_max_pred = int(y_max_pred * h)

    img_pred_bbox = bounding_boxes(image_bytes, x_min_pred, y_min_pred, x_max_pred, y_max_pred, class_id, 224, 224)
    
    test_table.add_data(img_ground_truth_bbox, img_pred_bbox)

run.log({"table_key": test_table})

