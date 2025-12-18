import base64
import os
import shutil
import threading
import queue
import time
from io import BytesIO

import cv2
import torch
import gc
import numpy as np
import imageio
from PIL import Image
from triton.language import dtype

from models.gdino.models.gdino import GDINO
from models.sam2.sam import SAM
from utils.color import COLOR
import pyrealsense2 as rs
from utils.utils import save_frames_to_temp_dir, get_object_iou, batch_mask_iou, batch_box_iou, \
    visualize_selected_masks_as_video, filter_mask_outliers
from utils.ObjectInfoManager import ObjectInfoManager

class Lang2SegTrack:
    def __init__(self, sam_type:str="sam2.1_hiera_tiny", model_path:str="models/sam2/checkpoints/sam2.1_hiera_large.pt",
                 video_path:str="", output_path:str="", use_txt_prompt:bool=False, max_frames:int=60,
                 first_prompts: list | None = None, save_video=True, device="cuda:0", mode="realtime"):
        self.sam_type = sam_type # the type of SAM model to use
        self.model_path = model_path # the path to the SAM model checkpoint
        self.video_path = video_path # the path to the video to track. If mode="video", this param is required.
        self.output_path = output_path # the path to save the output video. If save_video=False, this param is ignored.
        self.max_frames = max_frames # The maximum number of frames to be retained, beyond which the oldest frames are deleted,
        # so that the memory footprint does not grow indefinitely
        # If the number of tracked objects is large and likely to be occluded, set it to a larger value(such as 120) to enhance tracking
        self.first_prompts = first_prompts  # the initial bounding boxes ,points or masks to track. If not None, the tracker will use the first frame to detect objects.
        # [mask, point, bbox], mask: np.ndarray[H, W], point: list[int], bbox: list[int]
        self.save_video = save_video # whether to save the output video
        self.device = device
        self.mode = mode # the mode to run the tracker. "video" or "realtime"
        if self.mode == 'img' and not use_txt_prompt:
            raise ValueError("In 'img' mode, use_txt_prompt must be True")

        self.sam = SAM()
        self.sam.build_model(self.sam_type, self.model_path, predictor_type=mode, device=device, use_txt_prompt=use_txt_prompt)
        if use_txt_prompt:
            self.gdino = GDINO()
            self.gdino_16 = False
            if not self.gdino_16:
                print("Building GroundingDINO model...")
                self.gdino.build_model(device=device)
        else:
            self.gdino = None

        # data management
        self.history_frames = []
        self.all_forward_masks = {}
        self.all_final_masks = {}
        self.object_start_frame_idx = {}
        self.object_start_prompts = {}
        self.existing_obj_outputs = []
        self.current_text_prompt = None
        self.last_text_prompt = None
        if self.first_prompts is not None:
            self.prompts = {'prompts': self.first_prompts, 'labels': [None] * len(self.first_prompts), 'scores': [None] * len(self.first_prompts)}
            self.add_new = True
        else:
            self.prompts = {'prompts': [], 'labels': [], 'scores': []}
        self.iou_threshold = 0.3
        self.detection_frequency = 5

        self.input_queue = queue.Queue()
        self.drawing = False
        self.add_new = False
        self.ix, self.iy = -1, -1
        self.frame_display = None
        self.height, self.width = None, None
        self.prev_time = 0

    def input_thread(self):
        while True:
            user_input = input()
            self.input_queue.put(user_input)

    def draw_bbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.prompts['prompts'].append((x, y))
                self.prompts['labels'].append(None)
                self.prompts['scores'].append(None)
                self.add_new = True
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            else:
                self.drawing = True
                self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img = param.copy()
            cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Video Tracking", img)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            if abs(x - self.ix) > 2 and abs(y - self.iy) > 2:
                bbox = [self.ix, self.iy, x, y]
                self.prompts['prompts'].append(bbox)
                self.prompts['labels'].append(None)
                self.prompts['scores'].append(None)
                self.add_new = True
                cv2.rectangle(param, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.drawing = False

    def add_to_state(self, predictor, state, prompts, start_with_0=False):
        frame_idx = 0 if start_with_0 else state["num_frames"]-1
        for id, item in enumerate(prompts['prompts']):
            if len(item) == 4:
                x1, y1, x2, y2 = item
                cv2.rectangle(self.frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                predictor.add_new_points_or_box(state, box=item, frame_idx=frame_idx, obj_id=id)
            elif len(item) == 2:
                x, y = item
                cv2.circle(self.frame_display, (x, y), 5, (0, 255, 0), -1)
                pt = torch.tensor([[x, y]], dtype=torch.float32)
                lbl = torch.tensor([1], dtype=torch.int32)
                predictor.add_new_points_or_box(state, points=pt, labels=lbl, frame_idx=frame_idx, obj_id=id)
            else:
                predictor.add_new_mask(state, mask=item, frame_idx=frame_idx, obj_id=id)

    def track_and_visualize(self, predictor, state, frame, writer):
        if (any(len(state["point_inputs_per_obj"][i]) > 0 for i in range(len(state["point_inputs_per_obj"]))) or
            any(len(state["mask_inputs_per_obj"][i]) > 0 for i in range(len(state["mask_inputs_per_obj"])))):
            frame_idx, obj_ids, masks = predictor.propagate_in_frame(state, state["num_frames"] - 1)
            self.existing_obj_outputs = []
            # self.prompts['prompts'] = []
            for obj_id, mask in zip(obj_ids, masks):
                mask = mask[0].cpu().numpy() > 0.0
                mask = filter_mask_outliers(mask)
                nonzero = np.argwhere(mask)
                if nonzero.size == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = nonzero.min(axis=0)
                    y_max, x_max = nonzero.max(axis=0)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                self.draw_mask_and_bbox(frame, mask, bbox, obj_id)
                # self.existing_masks.append(mask)
                self.existing_obj_outputs.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                # self.prompts['prompts'].append(mask)
                # self.all_forward_masks.setdefault(obj_id, []).append(mask)
            self.prompts['prompts'] = self.existing_obj_outputs.copy()

        frame_dis = self.show_fps(frame)
        cv2.imshow("Video Tracking", frame_dis)

        if writer:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)


    def draw_mask_and_bbox(self, frame, mask, bbox, obj_id):
        mask_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_img[mask] = COLOR[obj_id % len(COLOR)]
        frame[:] = cv2.addWeighted(frame, 1, mask_img, 0.6, 0)
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR[obj_id % len(COLOR)], 2)
        # cv2.putText(frame, f"obj_{obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[obj_id % len(COLOR)], 2)


    def show_fps(self, frame):
        frame = frame.copy()
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        fps_str = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def visualize_final_masks(self, output_path="final_tracked_video.mp4", fps=25):
        if not hasattr(self, "all_final_masks") or not self.all_final_masks:
            print("No final masks found. Please run `track()` and `track_backward()` first.")
            return

        print("Visualizing final tracking results...")
        num_frames = len(self.all_final_masks[0])
        assert len(self.history_frames)== num_frames
        writer = imageio.get_writer(output_path, fps=fps)

        for frame_idx in range(num_frames):
            base_frame = self.history_frames[frame_idx].copy()
            for obj_id, mask_list in self.all_final_masks.items():
                mask = mask_list[frame_idx]
                nonzero = np.argwhere(mask)
                if nonzero.size == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = nonzero.min(axis=0)
                    y_max, x_max = nonzero.max(axis=0)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                self.draw_mask_and_bbox(base_frame, mask, bbox, obj_id)

            writer.append_data(cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB))
            cv2.imshow("Final Tracking Visualization", base_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        writer.close()
        print(f"Final visualization saved to {output_path}")
        cv2.destroyAllWindows()

    def predict_img(
            self,
            images_pil: list[Image.Image],
            texts_prompt: list[str],
            box_threshold: float = 0.3,
            text_threshold: float = 0.25,
    ):
        """
        Parameters:
            images_pil (list[Image.Image]): List of input images.
            texts_prompt (list[str]): List of text prompts corresponding to the images.
            box_threshold (float): Threshold for box predictions.
            text_threshold (float): Threshold for text predictions.
        Returns:
            list[dict]: List of results containing masks and other outputs for each image.
            Output format:
            [{
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }, ...]
        """
        if self.gdino_16:
            if len(images_pil) > 1:
                raise ValueError("GroundingDINO_16 only support single image")
            byte_io = BytesIO()
            images_pil[0].save(byte_io, format='PNG')
            image_bytes = byte_io.getvalue()
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            texts_prompt = texts_prompt[0]
            gdino_results = self.gdino.predict_dino_1_6_pro(base64_encoded, texts_prompt, box_threshold, text_threshold)
        else:
            gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            # print(f"Predicting {len(sam_boxes)} masks")
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
            # print(f"Predicted {len(all_results)} masks")
        return all_results

    def track_backward(self):
        predictor = self.sam.video_predictor

        print("Starting backward tracking for each object...")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for obj_id in range(max(self.object_start_frame_idx)+1):

                start_idx = self.object_start_frame_idx[obj_id]
                if start_idx == 0:
                    full_masks = self.all_forward_masks[obj_id]
                else:
                    print('\n')
                    print(f"\033[92mINFO: Object_{obj_id} is being tracked backward in time......\033[0m")
                    history_frames = self.history_frames[:start_idx]
                    history_frames = history_frames[::-1]
                    frames = save_frames_to_temp_dir(history_frames)
                    prompt = self.object_start_prompts[obj_id]
                    reverse_state = predictor.init_state(
                        frames, offload_state_to_cpu=False, offload_video_to_cpu=False
                    )
                    self.add_to_state(predictor, reverse_state, [prompt], start_with_0=True)
                    backward_masks = []
                    for frame_idx, obj_ids, masks in predictor.propagate_in_video(reverse_state):
                        for mid, mask in zip(obj_ids, masks):
                            mask_np = mask[0].cpu().numpy() > 0.0
                            backward_masks.append(mask_np)

                    backward_masks = backward_masks[::-1]
                    forward_masks = self.all_forward_masks.get(obj_id, [])
                    full_masks = backward_masks + forward_masks[1:] if len(forward_masks) > 1 else backward_masks
                    #predictor.reset_state(reverse_state)
                self.all_final_masks[obj_id] = full_masks

        print("Backward tracking completed. Merged object trajectories are ready.")

        # save mask img
        output_dir = "mask_outputs"
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for obj_id, masks in self.all_final_masks.items():
            obj_dir = os.path.join(output_dir, f"obj_{obj_id}")
            os.makedirs(obj_dir, exist_ok=True)
            for frame_idx, mask in enumerate(masks):
                mask_image = (mask * 255).astype(np.uint8)
                mask_path = os.path.join(obj_dir, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(mask_path, mask_image)
        print(f"Masks saved to {output_dir}")
        visualize_selected_masks_as_video()


    def track(self):

        predictor = self.sam.video_predictor

        if self.mode == "realtime":
            print("Start with realtime mode.")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            get_frame = lambda: np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
        elif self.mode == "video":
            print("Start with video mode.")
            cap = cv2.VideoCapture(self.video_path)
            ret, color_image = cap.read()
            get_frame = lambda: cap.read()
        else:
            raise ValueError("The mode is not supported in this method.")

        self.height, self.width = color_image.shape[:2]

        if self.save_video:
            writer = imageio.get_writer(self.output_path, fps=25)
        else:
            writer = None

        cv2.namedWindow("Video Tracking")

        threading.Thread(target=self.input_thread, daemon=True).start()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state_from_numpy_frames([color_image], offload_state_to_cpu=False, offload_video_to_cpu=False)
            while True:
                if self.mode == "realtime":
                    frame = get_frame()
                else:
                    ret, frame = get_frame()
                    if not ret:
                        break
                self.frame_display = frame.copy()
                # self.history_frames.append(frame)
                cv2.setMouseCallback("Video Tracking", self.draw_bbox, param=self.frame_display)

                if not self.input_queue.empty():
                    self.current_text_prompt = self.input_queue.get()
                    # self.persistent_text_prompts.add(text)

                if self.current_text_prompt is not None:
                    # combined_prompt = ".".join(self.persistent_text_prompts)
                    if (state['num_frames']-1) % self.detection_frequency == 0 or self.last_text_prompt is None:
                        detection = self.gdino.predict(
                            [Image.fromarray(frame)],
                            [self.current_text_prompt],
                            0.4, 0.25
                        )[0]
                        scores = detection['scores'].cpu().numpy()
                        labels = detection['labels']
                        boxes = detection['boxes'].cpu().numpy().tolist()
                        boxes_np = np.array(boxes, dtype=np.int32)
                        labels_np = np.array(labels)
                        scores_np = np.array(scores)
                        filter_mask = scores > 0.3
                        valid_boxes = boxes_np[filter_mask]
                        valid_labels = labels_np[filter_mask]
                        valid_scores = scores_np[filter_mask]


                        # filter_mask = detection['scores'] > 0.3
                        # # valid_boxes = np.array(detection['boxes'], dtype=np.int32)[filter_mask]
                        # valid_labels = np.array(detection['labels'])[filter_mask]
                        # valid_scores = detection['scores'][filter_mask]
                        # valid_masks = detection['masks'][filter_mask]

                        if self.last_text_prompt != self.current_text_prompt:
                            self.prompts['prompts'].extend(valid_boxes)
                            # self.prompts['prompts'].extend(valid_masks)
                            self.prompts['labels'].extend(valid_labels)
                            self.prompts['scores'].extend(valid_scores)
                            self.add_new = True
                        elif self.existing_obj_outputs:
                            # iou_matrix = batch_mask_iou(valid_masks, np.array(self.existing_masks))
                            iou_matrix = batch_box_iou(valid_boxes, np.array(self.existing_obj_outputs))
                            is_new = np.max(iou_matrix, axis=1) < self.iou_threshold
                            valid_boxes = valid_boxes[is_new]
                            valid_labels = valid_labels[is_new]
                            valid_scores = valid_scores[is_new]
                            # valid_masks = valid_masks[is_new]
                            if len(valid_boxes) > 0:
                                self.prompts['prompts'].extend(valid_boxes)
                                # self.prompts['prompts'].extend(valid_masks)
                                self.prompts['labels'].extend(valid_labels)
                                self.prompts['scores'].extend(valid_scores)
                                self.add_new = True
                    self.last_text_prompt = self.current_text_prompt

                if self.add_new:
                    existing_obj_ids = set(state["obj_ids"])
                    predictor.reset_state(state)
                    self.add_to_state(predictor, state, self.prompts)
                    current_obj_ids = set(state["obj_ids"])
                    newly_added_ids = current_obj_ids - existing_obj_ids

                predictor.append_frame_to_inference_state(state, frame)
                self.track_and_visualize(predictor, state, frame, writer)

                if self.add_new:
                    for obj_id in newly_added_ids:
                        self.object_start_frame_idx[obj_id] = state['num_frames'] - 1
                        # self.object_start_prompts[obj_id] = self.all_forward_masks[obj_id][0]
                    self.add_new = False

                if state["num_frames"] % self.max_frames == 0:
                    if len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
                        predictor.append_frame_as_cond_frame(state, state["num_frames"] - 2)
                    predictor.release_old_frames(state)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        if self.mode == "realtime":
            pipeline.stop()
        else:
            cap.release()
        # self.track_backward()
        # self.visualize_final_masks()
        if writer:
            writer.close()
        cv2.destroyAllWindows()
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    tracker = Lang2SegTrack(sam_type="sam2.1_hiera_tiny",
                            model_path="models/sam2/checkpoints/sam2.1_hiera_tiny.pt",
                            video_path="assets/car.mp4",
                            # video_path="assets/05_default_juggle.mp4",
                            output_path="forward_tracked_video.mp4",
                            mode="video",
                            save_video=True,
                            use_txt_prompt=True)
    tracker.current_text_prompt = 'car'
    tracker.track()
