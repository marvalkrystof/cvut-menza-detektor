# CVUT Menza Camera Scraper

Downloads CVUT cafeteria camera images and detects/tracks people using the
[YOLOv5-CrowdHuman model](https://github.com/MahenderAutonomo/yolov5-crowdhuman).

## Setup

```bash
uv sync
```

Download `crowdhuman_yolov5m.pt` from the model repository and place it in the
project root.

## Scrape an image

```bash
uv run python cam_scrape.py TECHNICKA 1
```

Supported cameras are `TECHNICKA` and `STUDENTSKY_DUM`.

## Detect people in an image

```bash
uv run python detector.py input.jpg output.jpg
```

## Detect and track people in a video

For combining images into a video you can use FFMPEG.

```bash
uv run python video_detector.py input.mp4 output.mp4
```

The video detector uses a lightweight IoU tracker to keep person IDs between
frames. Use `--confidence`, `--image-size`, `--iou-threshold`, and
`--max-missed-frames` to tune it.


## Disclaimer
This runs on the basic crowdhuman model, there is still much fine-tuning to be done, but for now I cannot be bothered.