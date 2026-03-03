import random

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from matplotlib import pyplot as plt
import logging
import torch

MIN_AREA = 100



def get_device_type() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        logging.warning("No GPU found, using CPU instead")
        return "cpu"


device_type = get_device_type()
DEVICE = torch.device(device_type)

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def draw_image(image_rgb, masks, xyxy, probs, labels):
    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    # Create class_id for each unique label
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in labels]

    # Add class_id to the Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks.astype(bool),
        confidence=probs,
        class_id=np.array(class_id),
    )
    annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    return annotated_image

def display_image_with_boxes(image, boxes, logits, phrases):

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    hex_colors = ['#209ce3', '#fecce6', '#ffe9a5', '#3dbc75']
    color_map = {}
    for box, logit, phrase in zip(boxes, logits, phrases):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit, 2)  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        if phrase not in color_map:

            available_colors = [color for color in hex_colors if color not in color_map.values()]
            if not available_colors:
                available_colors = hex_colors
            color_map[phrase] = random.choice(available_colors)
        color = color_map[phrase]

        # Draw bounding box with unique color and thicker line
        rect = plt.Rectangle(
            (x_min, y_min), box_width, box_height,
            fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)

        # Add confidence score and phrase as text
        ax.text(
            x_min, y_min - 5, f"{phrase}: {confidence_score}",
            fontsize=9, color=color, verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

    plt.show()


def save_image_with_boxes(image, boxes, logits, phrases, save_path, dpi=150):
    """Draw image with bounding boxes and save to file (same layout as display_image_with_boxes)."""
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    hex_colors = ['#209ce3', '#fecce6', '#ffe9a5', '#3dbc75']
    color_map = {}
    for box, logit, phrase in zip(boxes, logits, phrases):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(logit, 2)
        box_width = x_max - x_min
        box_height = y_max - y_min

        if phrase not in color_map:
            available_colors = [color for color in hex_colors if color not in color_map.values()]
            if not available_colors:
                available_colors = hex_colors
            color_map[phrase] = random.choice(available_colors)
        color = color_map[phrase]

        rect = plt.Rectangle(
            (x_min, y_min), box_width, box_height,
            fill=False, edgecolor=color, linewidth=2
        )
        ax.add_patch(rect)

        ax.text(
            x_min, y_min - 5, f"{phrase}: {confidence_score}",
            fontsize=9, color=color, verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def get_contours(mask):
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, 0)
    mask = mask.astype(np.uint8)
    mask *= 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    effContours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            effContours.append(c)
    return effContours


def contour_to_points(contour):
    pointsNum = len(contour)
    contour = contour.reshape(pointsNum, -1).astype(np.float32)
    points = [point.tolist() for point in contour]
    return points


def generate_labelme_json(binary_masks, labels, image_size, image_path=None):
    """Generate a LabelMe format JSON file from binary mask tensor.

    Args:
        binary_masks: Binary mask tensor of shape [N, H, W].
        labels: List of labels for each mask.
        image_size: Tuple of (height, width) for the image size.
        image_path: Path to the image file (optional).

    Returns:
        A dictionary representing the LabelMe JSON file.
    """
    num_masks = binary_masks.shape[0]
    binary_masks = binary_masks.numpy()

    json_dict = {
        "version": "4.5.6",
        "imageHeight": image_size[0],
        "imageWidth": image_size[1],
        "imagePath": image_path,
        "flags": {},
        "shapes": [],
        "imageData": None,
    }

    # Loop through the masks and add them to the JSON dictionary
    for i in range(num_masks):
        mask = binary_masks[i]
        label = labels[i]
        effContours = get_contours(mask)

        for effContour in effContours:
            points = contour_to_points(effContour)
            shape_dict = {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": points,
                "shape_type": "polygon",
            }

            json_dict["shapes"].append(shape_dict)

    return json_dict
