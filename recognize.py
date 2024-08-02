import threading
import time
import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector (choose one)
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# Face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

# Load precomputed face features and names
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# Mapping of face IDs to names
id_face_mapping = {}

# Data mapping for tracking information
data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
}

# Flag to indicate end of video processing
video_end_flag = threading.Event()

def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def process_tracking(frame, detector, tracker, args, frame_id, fps):
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = tracker.update(outputs, [img_info["height"], img_info["width"]], (128, 128))
        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)
        tracking_image = plot_tracking(
            img_info["raw_img"], tracking_tlwhs, tracking_ids, names=id_face_mapping, frame_id=frame_id + 1, fps=fps
        )
    else:
        tracking_image = img_info["raw_img"]

    data_mapping["raw_image"] = img_info["raw_img"]
    data_mapping["detection_bboxes"] = bboxes
    data_mapping["detection_landmarks"] = landmarks
    data_mapping["tracking_ids"] = tracking_ids
    data_mapping["tracking_bboxes"] = tracking_bboxes

    return tracking_image

@torch.no_grad()
def get_feature(face_image):
    face_preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((112, 112)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    emb_img_face = recognizer(face_image).cpu().numpy()
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb

def recognition(face_image):
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]
    return score, name

def mapping_bbox(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area
    return iou

def tracking(detector, args):
    start_time = time.time_ns()
    frame_count = 0
    fps = -1
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    cap = cv2.VideoCapture('video.mp4')

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        _, img = cap.read()
        if img is None:
            break

        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)

        # Draw names prominently
        for tid, bbox in zip(data_mapping["tracking_ids"], data_mapping["tracking_bboxes"]):
            if tid in id_face_mapping:
                name = id_face_mapping[tid]
                x1, y1, x2, y2 = map(int, bbox)  # Ensure coordinates are integers
                cv2.rectangle(tracking_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Calculate the width of the text to determine the size of the black background
                (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(tracking_image, (x1-5, y1 - text_height - 15), (x1 + text_width + 5, y1), (0, 0, 0), -1)
                cv2.putText(tracking_image, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()

        cv2.imshow("Face Recognition", tracking_image)

        # Write frame to video file
        out.write(tracking_image)

        # Check for user exit input
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Signal that video processing has ended
    video_end_flag.set()

def recognize():
    while not video_end_flag.is_set():
        raw_image = data_mapping["raw_image"]
        detection_landmarks = data_mapping["detection_landmarks"]
        detection_bboxes = data_mapping["detection_bboxes"]
        tracking_ids = data_mapping["tracking_ids"]
        tracking_bboxes = data_mapping["tracking_bboxes"]

        if not tracking_bboxes:
            print("Waiting for a person...")
            time.sleep(1)  # Add a small sleep to reduce CPU usage
            continue

        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                if mapping_score > 0.9:
                    face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])
                    score, name = recognition(face_image=face_alignment)
                    if name is not None:
                        if score < 0.25:
                            caption = "UN_KNOWN"
                        else:
                            caption = f"{name}:{score:.2f}"
                    id_face_mapping[tracking_ids[i]] = caption
                    detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)
                    break

def main():
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    thread_track = threading.Thread(target=tracking, args=(detector, config_tracking))
    thread_track.start()

    thread_recognize = threading.Thread(target=recognize)
    thread_recognize.start()

    # Wait for threads to finish
    thread_track.join()
    video_end_flag.set()  # Ensure recognize thread stops
    thread_recognize.join()

if __name__ == "__main__":
    main()
