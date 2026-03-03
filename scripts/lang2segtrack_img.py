from typing import Any


import base64
import os
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
from models.gdino.models.gdino import GDINO
from models.gdino.utils import display_image_with_boxes, save_image_with_boxes
from models.sam2.sam import SAM
from utils.color import COLOR
import pyrealsense2 as rs

from utils.utils import filter_mask_outliers


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

        self.input_queue = queue.Queue()
        self.drawing = False
        self.add_new = False
        self.ix, self.iy = -1, -1
        self.frame_display = None
        self.height, self.width = None, None
        if self.first_prompts is not None:
            self.prompts_list = self.first_prompts
            self.add_new = True
        else:
            self.prompts_list = []
        self.prev_time = 0

    def input_thread(self):
        while True:
            user_input = input()
            self.input_queue.put(user_input)

    def draw_bbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.prompts_list.append((x, y))
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
                self.prompts_list.append(bbox)
                self.add_new = True
                cv2.rectangle(param, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.drawing = False

    def add_to_state(self, predictor, state, list):
        for id, item in enumerate(list):
            if len(item) == 4:
                x1, y1, x2, y2 = item
                cv2.rectangle(self.frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                predictor.add_new_points_or_box(state, box=item, frame_idx=state["num_frames"] - 1, obj_id=id)
            elif len(item) == 2:
                x, y = item
                cv2.circle(self.frame_display, (x, y), 5, (0, 255, 0), -1)
                pt = torch.tensor([[x, y]], dtype=torch.float32)
                lbl = torch.tensor([1], dtype=torch.int32)
                predictor.add_new_points_or_box(state, points=pt, labels=lbl, frame_idx=state["num_frames"] - 1, obj_id=id)
            else:
                predictor.add_new_mask(state, mask=item, frame_idx=state["num_frames"] - 1, obj_id=id)

    def track_and_visualize(self, predictor, state, frame, writer):
        if (any(len(state["point_inputs_per_obj"][i]) > 0 for i in range(len(state["point_inputs_per_obj"]))) or
            any(len(state["mask_inputs_per_obj"][i]) > 0 for i in range(len(state["mask_inputs_per_obj"])))):
            frame_idx, obj_ids, masks = predictor.propagate_in_frame(state, state["num_frames"] - 1)
            self.prompts_list=[]
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
                self.prompts_list.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
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
        # cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR[obj_id % len(COLOR)], 2)
        # cv2.putText(frame, f"obj_{obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[obj_id % len(COLOR)], 2)


    def show_fps(self, frame):
        frame = frame.copy()
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        fps_str = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return frame


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
        if self.mode != "img":
            raise ValueError("This method only support use 'img' mode")
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
                cv2.setMouseCallback("Video Tracking", self.draw_bbox, param=self.frame_display)

                if not self.input_queue.empty():
                    text = self.input_queue.get()
                    out = self.gdino.predict([Image.fromarray(frame)], [text], 0.3, 0.25)
                    boxes = [[int(v) for v in box] for box in out[0]["boxes"].cpu().numpy().tolist()]
                    self.prompts_list.extend(boxes)
                    self.add_new = True

                if self.add_new:
                    predictor.reset_state(state)
                    self.add_to_state(predictor, state, self.prompts_list)
                    self.add_new = False

                predictor.append_frame_to_inference_state(state, frame)
                self.track_and_visualize(predictor, state, frame, writer)
                if state["num_frames"] % self.max_frames == 0:
                    if len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
                        predictor.append_frame_as_cond_frame(state, state["num_frames"] - 2)
                    predictor.release_old_frames(state)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.mode == "realtime":
            pipeline.stop()
        else:
            cap.release()
        if writer:
            writer.close()
        cv2.destroyAllWindows()
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    mask = Image.open("mask_images/mask_0.png")
    mask = np.array(mask)
    tracker = Lang2SegTrack(sam_type="sam2.1_hiera_tiny",
                            model_path="models/sam2/checkpoints/sam2.1_hiera_tiny.pt",
                            video_path="assets/05_default_juggle.mp4",
                            output_path="processed_video.mp4",
                            mode="img",
                            # first_prompts=[mask],
                            save_video=True,
                            use_txt_prompt=True)
    # tracker.track()

    out = tracker.predict_img(
        [Image.open("assets/frame_00000.jpg")],
        ["cup"],
    )
    print(out)
    img = cv2.imread("assets/frame_00000.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # display_image_with_boxes(img, list(out[0]["boxes"]), out[0]["scores"], list(out[0]["labels"]))
    save_image_with_boxes(img, list(out[0]["boxes"]), out[0]["scores"], list[Any](out[0]["labels"]), "output/boxes.png")