"""Detect and track people in a video with YOLOv5-CrowdHuman."""

import argparse
import logging
from pathlib import Path

import cv2

from detector import WEIGHTS_PATH, load_model


logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("out.mp4")
IMAGE_SIZE = 640
CONFIDENCE = 0.05
IOU_THRESHOLD = 0.25
MAX_MISSED_FRAMES = 10


def intersection_over_union(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


class IoUTracker:
    def __init__(self, iou_threshold=IOU_THRESHOLD, max_missed_frames=MAX_MISSED_FRAMES):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        unmatched_tracks = set(self.tracks)
        assignments = []
        candidates = sorted(detections, key=lambda detection: detection[4], reverse=True)

        for detection in candidates:
            best_id = None
            best_iou = self.iou_threshold
            for track_id in unmatched_tracks:
                overlap = intersection_over_union(
                    detection[:4], self.tracks[track_id]["box"]
                )
                if overlap > best_iou:
                    best_iou, best_id = overlap, track_id

            if best_id is not None:
                self.tracks[best_id]["box"] = detection[:4]
                self.tracks[best_id]["missed"] = 0
                unmatched_tracks.remove(best_id)
                assignments.append((best_id, detection))

        for track_id in unmatched_tracks:
            self.tracks[track_id]["missed"] += 1

        assigned = {id(detection) for _, detection in assignments}
        for detection in candidates:
            if id(detection) not in assigned:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {"box": detection[:4], "missed": 0}
                assignments.append((track_id, detection))

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["missed"] <= self.max_missed_frames
        }
        return assignments


def open_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    return capture, width, height, fps


def draw_tracks(frame, assignments):
    for track_id, detection in assignments:
        x1, y1, x2, y2, confidence, _ = detection
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"person #{track_id} {confidence:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def process_video(model, input_path, output_path, image_size):
    capture, width, height, fps = open_video(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create video: {output_path}")

    tracker = IoUTracker()
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model(frame, size=image_size)
            detections = [
                detection
                for detection in results.xyxy[0].cpu().numpy()
                if int(detection[5]) == 0
            ]
            draw_tracks(frame, tracker.update(detections))
            writer.write(frame)

            frame_number += 1
            if frame_number % 50 == 0:
                logger.info("Processed %d frames", frame_number)
    finally:
        capture.release()
        writer.release()

    logger.info("Processed %d frames", frame_number)
    logger.info("Saved tracked video to %s", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("output_positional", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, dest="output_option")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--max-missed-frames", type=int, default=MAX_MISSED_FRAMES)
    args = parser.parse_args()
    args.output = args.output_option or args.output_positional
    if args.output is None:
        args.output = Path("data/processed/video") / f"{args.video.stem}_tracked.mp4"
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    model = load_model(args.weights)
    model.conf = args.confidence
    process_video(model, args.video, args.output, args.image_size)


if __name__ == "__main__":
    main()
