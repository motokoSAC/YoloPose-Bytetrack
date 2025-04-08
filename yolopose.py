import cv2 as cv
import numpy as np
from ultralytics import YOLO
from queue import Queue
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from util import match_bounding_boxes


class YoloByteCam:
    def __init__(self, input_stream):
        self.input_stream = input_stream
        self.track_img_queue = Queue(maxsize=1)
        self.pose_img_queue = Queue(maxsize=1)
        self.bbox_result_queue = Queue(maxsize=1)
        self.model_path = "modelzoo/yolo11n-pose.pt"
        self.model_track = YOLO(self.model_path)
        self.model_infer = YOLO(self.model_path)
        self.cap = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.running = False
        self.track_ids = {}
        self.current_track_id = None
        self.current_index = 0
        self.frame_shape = [720, 1280]
        self.calculate_fps = True
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()
        self.pose_kpts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.desired_fps = 60

    def start(self):
        self.running = True
        # start 3 thread
        self.executor.submit(self.camera)
        self.executor.submit(self.cam_infer)
        self.executor.submit(self.cam_track)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.executor.shutdown(wait=True)
        cv.destroyAllWindows()

    def camera(self):
        self.cap = cv.VideoCapture(self.input_stream)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_shape[1])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_shape[0])
        self.cap.set(cv.CAP_PROP_FPS, self.desired_fps)
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Video has ended or cannot be read.")
                break
            if not self.pose_img_queue.full() and not self.track_img_queue.full():
                self.track_img_queue.put(frame)
                self.pose_img_queue.put(frame)
            else:
                time.sleep(0.001)
        self.cap.release()
        self.running = False

    def cam_infer(self):
        while self.running:
            if not self.pose_img_queue.empty() and not self.bbox_result_queue.empty():
                frame = self.pose_img_queue.get()
                bytetrack_bbox_result = self.bbox_result_queue.get()
            else:
                time.sleep(0.001)
                continue

            if not bytetrack_bbox_result:
                print("bytetrack_bbox_result is none")

            orig_img = frame
            results = self.model_infer.predict(source=orig_img, conf=0.3, retina_masks=False, verbose=False, max_det=10)
            self.track_ids.clear()

            if results is not None:
                for i, result in enumerate(results):
                    if result.keypoints is None:
                        continue
                    pred_kpts = result.keypoints.xy
                    try:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        for index, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            person_id = match_bounding_boxes(box, bytetrack_bbox_result)
                            self.track_ids.setdefault(index, person_id)
                            cv.putText(frame, f"bytetrackID: {person_id} - poseID:{index}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    except AttributeError as e:
                        print(f'cam_infer error: {e}')

                    for person, kpts in enumerate(pred_kpts):
                        keypoints = kpts.cpu().numpy()
                        for kp_id, keypoint in enumerate(keypoints):
                            x, y = keypoint[:2]
                            if kp_id not in self.pose_kpts:
                                continue

                            if np.isnan(x) or np.isnan(y):
                                pxl_x = 0
                                pxl_y = 0
                            else:
                                pxl_x = x
                                pxl_y = y
                            pxl_x = int(round(pxl_x))
                            pxl_y = int(round(pxl_y))
                            cv.circle(frame, (pxl_x, pxl_y), 3, (0, 0, 255), -1)

            curr_time = time.time()

            if self.calculate_fps:
                self.frame_count += 1
                if (curr_time - self.prev_time) > 1.0:
                    self.fps = self.frame_count / (curr_time - self.prev_time)
                    self.prev_time = curr_time
                    self.frame_count = 0
                    cv.putText(frame, f"FPS: {self.fps:.2f}", (10, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
                else:
                    cv.putText(frame, f"FPS: {self.fps:.2f}", (10, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)

            cv.putText(frame, f"track-id:{self.current_track_id}", (10, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
            cv.imshow(f'cam{self.input_stream}', frame)
            k = cv.waitKey(1)

            if k & 0xFF == 27:
                self.stop()
                break
        cv.destroyAllWindows()

    def cam_track(self):
        while self.running:
            if not self.track_img_queue.empty():
                frame = self.track_img_queue.get()
            else:
                time.sleep(0.001)
                continue

            orig_img = frame
            result = self.model_track.track(source=orig_img, conf=0.3, iou=0.7, persist=True, verbose=False, tracker='cfg/custom_bytetrack.yaml')
            try:
                if result is not None:
                    boxes = result[0].boxes.xyxy.cpu().numpy()
                    ids = result[0].boxes.id.cpu().numpy()
                    if not self.bbox_result_queue.full():
                        id_box_dict = {id: box for id, box in zip(ids, boxes)}
                        self.bbox_result_queue.put(id_box_dict)
                    else:
                        time.sleep(0.001)
                        continue
                else:
                    if not self.bbox_result_queue.full():
                        id_box_dict = {}
                        self.bbox_result_queue.put(id_box_dict)
                    else:
                        time.sleep(0.001)
                        continue

            except AttributeError as e:  # Usually, people are not recognized
                print(f'cam_track error: {e}')
                id_box_dict = {}
                self.bbox_result_queue.put(id_box_dict)
                continue
            k = cv.waitKey(1)
            if k & 0xFF == 27:
                break
        cv.destroyAllWindows()

def run_camera_process(input_stream):
    cam = YoloByteCam(input_stream=input_stream)
    cam.start()

def main():
    streams = [
        1
    ]
    p = multiprocessing.Process(target=run_camera_process, args=(streams[0],))
    p.start()
    p.join()

if __name__ == '__main__':
    main()