"""Run the YOLOv5-CrowdHuman person/head detector."""

from pathlib import Path
import argparse

import cv2
import torch

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


WEIGHTS_PATH = Path("crowdhuman_yolov5m.pt")


def load_model(weights_path: Path) -> torch.nn.Module:
    """Load the YOLOv5-CrowdHuman model."""

    logger.info(f"Loading model from {weights_path}")
    if not weights_path.exists():
            logger.error(f"File not found: {weights_path}")
            raise FileNotFoundError(
                f"Missing {weights_path}. Download the CrowdHuman weights from "
                "https://github.com/MahenderAutonomo/yolov5-crowdhuman"
            )
        
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=str(weights_path),
        trust_repo=True,
    )
    return model
        
            

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image")
    parser.add_argument("output_positional", type=Path, nargs="?", help="Output image")
    parser.add_argument("--output", type=Path, dest="output_option")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)

    args = parser.parse_args()
    args.output = args.output_option or args.output_positional
    if args.output is None:
        args.output = Path("data/processed/images") / f"{args.image.stem}_processed.jpg"

    model = load_model(args.weights)
    model.conf = 0.05

    results = model(str(args.image), size=960)
    detections = results.xyxy[0].cpu().numpy()
    names = results.names

    person_count = sum(int(class_id) == 0 for *_, class_id in detections)
    head_count = sum(int(class_id) == 1 for *_, class_id in detections)
    logger.info(f"Detected people: {person_count}")
    logger.info(f"Detected heads: {head_count}")

    image = cv2.imread(str(args.image))
    if image is None:
        logger.error(f"Could not read {args.image}")
        raise FileNotFoundError(f"Could not read {args.image}")

    for x1, y1, x2, y2, confidence, class_id in detections:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"{names[int(class_id)]} {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image)
    logger.info(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    main()
