import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
import glob
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from tensorflow.keras.layers import InputLayer, Rescaling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, \
                        GlobalAveragePooling2D, SeparableConv2D
import tensorflow as tf
from keras.applications import VGG16, VGG19
from keras import backend as K
from zipfile import ZipFile as zf, BadZipFile
import cv2

# pip install git+https://github.com/openai/CLIP.git 

import torch
from torchvision import transforms
import clip  # this loads OpenAI's CLIP model
import joblib
from torch import nn


def clean_up():
    folder_path = "purple_cells"  # or any folder name
    files = glob.glob(f"{folder_path}/*")
    for file in files:
        os.remove(file)

def normalize_lighting(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h_, w_ = v.shape
    corners = np.concatenate([
        v[0:50, 0:50].flatten(),
        v[0:50, w_-50:w_].flatten(),
        v[h_-50:h_, 0:50].flatten(),
        v[h_-50:h_, w_-50:w_].flatten()
    ])
    bg_val = int(np.median(corners))
    target_bg = 230
    #print(bg_val)
    if bg_val < target_bg - 10:
        delta = target_bg - bg_val
    else:
        delta = 0  # no need to brighten if already bright

    v_int = v.astype(np.int16)
    v_new = np.clip(v_int + delta, 0, 255).astype(np.uint8)

    hsv_normalized = cv2.merge([h, s, v_new])
    return cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)

def fn_patching(image_path):
    image = cv2.imread(image_path)
    image = normalize_lighting(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Color detection with slightly widened ranges
    mask1 = cv2.inRange(hsv, np.array([115, 35, 35]), np.array([140, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([115, 40, 70]), np.array([175, 150, 255]))
    main_mask = cv2.bitwise_or(mask1, mask2)

    # Strict exclusion for bright non-cells
    exclude_lower = np.array([118, 20, 220])
    exclude_upper = np.array([130, 70, 255])
    exclude_mask = cv2.inRange(hsv, exclude_lower, exclude_upper)
    mask = cv2.bitwise_and(main_mask, cv2.bitwise_not(exclude_mask))

    # Morphological processing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Watershed segmentation
    contours_pre, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    wshed_img = image.copy()
    markers = cv2.watershed(wshed_img, markers)
    wshed_mask = np.zeros_like(mask)
    wshed_mask[markers > 1] = 255

    contours_post, _ = cv2.findContours(wshed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initial contour filtering
    final_contours = []
    for contour in contours_post:
        if len(contour) < 5:
            continue

        area = cv2.contourArea(contour)
        if area < 400 or area > 100000:
            #print(area)
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < 0.2:
            #print(circularity)
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.5 or aspect_ratio > 3.0:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        mask_roi = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_roi, [contour], -1, 255, -1)
        mean_val = cv2.mean(image, mask=mask_roi)
        blue, green, red = mean_val[:3]
        if not (blue > red + 8 and blue > green - 15):
            continue

        final_contours.append(contour)

    # Add back any valid pre-watershed contours
    for c in contours_pre:
        if len(c) < 5:
            continue

        area = cv2.contourArea(c)
        if area < 600 or area > 120000:
            #if area > 80000:
                #print(area)
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < 0.4:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.8:
            #print(aspect_ratio)
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            #print("hull")
            continue

        mask_roi = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_roi, [c], -1, 255, -1)
        mean_val = cv2.mean(image, mask=mask_roi)
        blue, green, red = mean_val[:3]
        if not (blue > red + 8 and blue > green - 15):
            continue

        unique = True
        for existing in final_contours:
            if cv2.matchShapes(existing, c, cv2.CONTOURS_MATCH_I2, 0) < 0.5:
                unique = False
                break
        if unique:
            final_contours.append(c)

    # Bounding box containment handling
    filtered_boxes = []
    boxes = [cv2.boundingRect(c) for c in final_contours]
    contours = list(zip(boxes, final_contours))

    contours.sort(key=lambda x: x[0][2]*x[0][3], reverse=True)

    while contours:
        current_box, current_contour = contours.pop(0)
        x1, y1, w1, h1 = current_box
        area1 = w1 * h1

        keep_current = True
        contained_contours = []

        for other_box, other_contour in contours[:]:
            x2, y2, w2, h2 = other_box
            area2 = w2 * h2

            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)

            if x_right < x_left or y_bottom < y_top:
                continue

            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            if intersection_area >= 0.9 * area2:
                contained_contours.append(other_contour)
                contours.remove((other_box, other_contour))

            elif intersection_area >= 0.9 * area1:
                keep_current = False
                break

        if keep_current:
            if contained_contours:
                if len(contained_contours) >= 1:
                    filtered_boxes.extend([(cv2.boundingRect(c), c) for c in contained_contours])
                else:
                    filtered_boxes.append((cv2.boundingRect(contained_contours[0]), contained_contours[0]))
            else:
                filtered_boxes.append((current_box, current_contour))

    final_contours = [contour for (box, contour) in filtered_boxes]

    # Output results
    os.makedirs("purple_cells", exist_ok=True)
    annotated = image.copy()
    cell_num = 0
    scale = 1.3

    for contour in final_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_x = max(cx - new_w // 2, 0)
        new_y = max(cy - new_h // 2, 0)
        end_x = min(new_x + new_w, image.shape[1])
        end_y = min(new_y + new_h, image.shape[0])

        cell = image[new_y:end_y, new_x:end_x]
        cv2.imwrite(f"purple_cells/wbc_{cell_num}.png", cell)
        cv2.rectangle(annotated, (new_x, new_y), (end_x, end_y), (0, 255, 0), 2)
        cell_num += 1

    #output_dir = "/content/annotated_results"
    #os.makedirs(output_dir, exist_ok=True)
    #filename = os.path.basename(image_path)
    #output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + "_annotated.png")

    output_path = "output.png"
    cv2.imwrite(output_path, annotated)
    return cell_num

class VLMClassifier(nn.Module):
    def __init__(self, image_feature_dim=512, tabular_dim=4, hidden_dim=128, num_classes=4):
        super().__init__()
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, tabular_features):
        tabular_out = self.tabular_net(tabular_features)
        combined = torch.cat([image_features, tabular_out], dim=1)
        return self.classifier(combined)

def predict_from_csv_and_return_results(csv_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load label encoder and model
    label_encoder = joblib.load("label_encoder.pkl")
    num_classes = len(label_encoder.classes_)

    vlm_model = VLMClassifier(num_classes=num_classes).to(device)
    vlm_model.load_state_dict(torch.load("best_vlm_model.pth", map_location=device))
    vlm_model.eval()

    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Validate structure
    if df.shape[0] != 1:
        raise ValueError("CSV must have exactly one data row (second row)")

    # Image preprocessing
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    row = df.iloc[0]
    image_path = row["Image"]
    img = Image.open(image_path).convert("RGB")
    img_tensor = image_transform(img).unsqueeze(0).to(device)

    # Encode image using CLIP
    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor).float()

        # Prepare tabular tensor
        tabular_tensor = torch.tensor([[row["Count"], row["Mean"], row["Max"], row["Std"]]],
                                      dtype=torch.float).to(device)

        # Predict
        output = vlm_model(image_features, tabular_tensor)
        pred_class_idx = torch.argmax(output, dim=1).item()
        pred_class_name = label_encoder.inverse_transform([pred_class_idx])[0]

    return pred_class_name

def preprocess_patch(path, target_size):
    img = image.load_img(path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def process_blood_smear(image_path, patch_path='purple_cells', model_eff_path='efficientnetv2_leukemia.keras', model_inc_path='inceptionv3_leukemia.keras'):
    # Load both models
    model_eff = load_model(model_eff_path)
    model_inc = load_model(model_inc_path)

    size = [299, 299]
    data = []

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Provided path is not a valid file: {image_path}")

    patch_preds = []
    clean_up()  # Clear previous patch data

    abs_image_path = os.path.abspath(image_path)
    print(f"Processing: {abs_image_path}")

    fn_patching(abs_image_path)

    image_files = [f for f in os.listdir(patch_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = len(image_files)
    if count == 0:
        print("No patches found.")
        return pd.DataFrame()

    for image_file in image_files:
        patch_file = os.path.join(patch_path, image_file)

        # Preprocess
        img_eff = preprocess_patch(patch_file, size)
        img_inc = preprocess_patch(patch_file, size)

        # Predict
        prob_eff = model_eff.predict(img_eff, verbose=0)
        prob_inc = model_inc.predict(img_inc, verbose=0)

        # Ensemble
        ensemble_prob = (prob_eff + prob_inc) / 2
        patch_preds.append(ensemble_prob)

    patch_preds = np.array(patch_preds)
    mean_score = np.mean(patch_preds)
    max_score = np.max(patch_preds)
    std_score = np.std(patch_preds)

    data.append({
        'Actual': os.path.basename(os.path.dirname(abs_image_path)),
        'Image': abs_image_path,
        'Count': count,
        'Mean': mean_score,
        'Max': max_score,
        'Std': std_score
    })

    # Save and return
    df = pd.DataFrame(data)
    df.to_csv('Data.csv', index=False)
    result = predict_from_csv_and_return_results('Data.csv')
    print(f"Predicted class: {result}")

    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
    else:
        image_path = sys.argv[1]
        result = process_blood_smear(image_path)
        print(f"Predicted class: {result}")