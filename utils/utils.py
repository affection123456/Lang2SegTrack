import os
import tempfile

import cv2
import imageio
import numpy as np
from PIL import Image

from utils.color import COLOR


def visualize_selected_masks_as_video(selected_obj_ids: list = [], masks_dir: str = "mask_outputs",
                                      output_video_path: str = "selected_masks_video.mp4",
                                      fps: int = 25):

    if not os.path.exists(masks_dir):
        print(f"Mask output folder ‘{masks_dir}’ does not exist.")
        return
    if not selected_obj_ids:
        obj_dirs = [d for d in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, d)) and d.startswith("obj_")]
        selected_obj_ids = [int(d.split("_")[1]) for d in obj_dirs]
        selected_obj_ids.sort()


    all_mask_paths = []
    max_frame_count = 0
    for obj_id in selected_obj_ids:
        obj_dir = os.path.join(masks_dir, f"obj_{obj_id}")
        if not os.path.exists(obj_dir):
            print(f"The subfolder ‘{obj_dir}’ of the object {obj_id} does not exist, skip the object.")
            continue
        mask_paths = sorted([os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith('.png')])
        all_mask_paths.append(mask_paths)
        max_frame_count = max(max_frame_count, len(mask_paths))

    if not all_mask_paths:
        print("No valid mask image is found, and the video cannot be generated.")
        return

    writer = imageio.get_writer(output_video_path, fps=fps)

    for frame_idx in range(max_frame_count):
        combined_mask = None
        for obj_idx, mask_paths in enumerate(all_mask_paths):
            if frame_idx < len(mask_paths):
                mask_path = mask_paths[frame_idx]
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if combined_mask is None:
                    combined_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                color_mask[mask > 0] = COLOR[selected_obj_ids[obj_idx] % len(COLOR)]
                combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 0.6, 0)

        if combined_mask is not None:
            writer.append_data(combined_mask)
            cv2.imshow("Selected Masks Visualization", combined_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    writer.close()
    cv2.destroyAllWindows()
    print(f"The mask trace video of the selected object has been saved to ‘{output_video_path}’.")

def batch_mask_iou(masks1, masks2):

    masks1 = masks1.astype(bool)
    masks2 = masks2.astype(bool)

    area1 = np.sum(masks1, axis=(1, 2))
    area2 = np.sum(masks2, axis=(1, 2))

    intersection = np.sum(
        masks1[:, np.newaxis, ...] & masks2[np.newaxis, ...],
        axis=(2, 3)
    )

    union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection

    iou_matrix = np.where(union > 0, intersection / union, 0.0)

    return iou_matrix


def batch_box_iou(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    xmin = np.maximum(x11, x21.T)
    ymin = np.maximum(y11, y21.T)
    xmax = np.minimum(x12, x22.T)
    ymax = np.minimum(y12, y22.T)

    w = np.maximum(0, xmax - xmin)
    h = np.maximum(0, ymax - ymin)
    inter = w * h
    iou = inter / (area1 + area2.T - inter)
    return iou


def get_object_iou(new_bbox, existing_bbox):
    x1, y1, x2, y2 = new_bbox
    x1e, y1e, x2e, y2e = existing_bbox

    inter_x1 = max(x1, x1e)
    inter_y1 = max(y1, y1e)
    inter_x2 = min(x2, x2e)
    inter_y2 = min(y2, y2e)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_new = (x2 - x1) * (y2 - y1)
    area_existing = (x2e - x1e) * (y2e - y1e)
    union_area = area_new + area_existing - inter_area

    return inter_area / union_area if union_area > 0 else 0


def filter_mask_outliers(mask, min_area_ratio=0.01, min_size=5):

    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_morph = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_morph, connectivity=8)

    if num_labels == 1:
        return np.zeros_like(mask, dtype=bool)

    valid_labels = [i for i in range(1, num_labels) if stats[i, 4] > min_size]
    if not valid_labels:
        return np.zeros_like(mask, dtype=bool)

    areas = [stats[i, 4] for i in valid_labels]
    max_area_idx = np.argmax(areas)
    max_area = areas[max_area_idx]

    filtered_mask = np.zeros_like(mask, dtype=bool)
    for label in valid_labels:
        area = stats[label, 4]
        if area / max_area >= min_area_ratio:
            filtered_mask[labels == label] = True

    return filtered_mask

def bbox_process(bbox_list, labels=None):
    prompts = {}
    for fid, bbox in enumerate(bbox_list):
        x1, y1, x2, y2 = bbox
        label = labels[fid] if labels else f"obj_{fid}"
        prompts[fid] = ((int(x1), int(y1), int(x2), int(y2)), label)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        line = line.strip()
        if len(line) == 0:
            continue
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or os.path.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def save_frames_to_temp_dir(frames: list[np.ndarray]) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="chunk_frames_")
    for i, frame in enumerate(frames):
        path = os.path.join(tmp_dir, f"{i:04d}.jpg")
        # OpenCV uses BGR, PIL expects RGB
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)
    return tmp_dir


def extract_frames(
        video_path,
        output_dir,
        frame_ids=None,
        frame_range=None):

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    selected_frames = set()
    if frame_ids:
        selected_frames.update([i for i in frame_ids if 0 <= i < total_frames])
    if frame_range:
        start, end = frame_range
        selected_frames.update(range(max(0, start), min(end + 1, total_frames)))

    selected_frames = sorted(selected_frames)
    print(f"Extracting frames: {selected_frames}")

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in selected_frames:
            filename = os.path.join(output_dir, f"frame_{current_frame:05d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved frame {current_frame} to {filename}")

        current_frame += 1
        if current_frame > max(selected_frames):
            break

    cap.release()
    print("Done.")

extract_frames('assets/04_coffee.mp4',
               'assets',
               [0])