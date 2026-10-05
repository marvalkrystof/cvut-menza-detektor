from enum import Enum

import requests
import time
import argparse

import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_offline_image(content):
  """Return True when the camera returns the red OFFLINE placeholder."""
  encoded = np.frombuffer(content, dtype=np.uint8)
  image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
  if image is None:
    return False

  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  red = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) &
         (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 100))
  red_pixels = np.argwhere(red)
  if len(red_pixels) == 0:
    return False

  red_ratio = len(red_pixels) / red.size
  y_min, x_min = red_pixels.min(axis=0)
  y_max, x_max = red_pixels.max(axis=0)
  red_width = x_max - x_min + 1
  red_height = y_max - y_min + 1
  dark_ratio = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 30).mean()

  # The actual placeholder is almost entirely black with a centered red
  # OFFLINE label. Normal camera frames are grayscale.
  return (
    len(red_pixels) >= 500 and
    red_ratio >= 0.002 and
    red_width >= image.shape[1] * 0.15 and
    red_height >= image.shape[0] * 0.03 and
    dark_ratio >= 0.70
  )


class Menza(Enum):
  TECHNICKA = "technicka"
  STUDENTSKY_DUM = "studentsky-dum"

  @staticmethod
  def get_menza_url(menza):
    urls = {
      Menza.TECHNICKA: "https://agata.suz.cvut.cz/jidelnicky/tm-cam-img.php?x=1",
      Menza.STUDENTSKY_DUM: "https://agata.suz.cvut.cz/jidelnicky/sd-cam-img.php?x=2"
    }
    return urls[menza]


def parse_menza(value):
  normalized = value.upper().replace("-", "_")
  try:
    return Menza[normalized]
  except KeyError as error:
    raise argparse.ArgumentTypeError(
      "choose TECHNICKA or STUDENTSKY_DUM"
    ) from error


def main():
    parser = argparse.ArgumentParser("cam_scrape.py", description="Scrapes images from a camera feed and saves them to a local directory.")
    parser.add_argument(
    "menza",
    help="The menza to scrape images from.",
    type=parse_menza,
    metavar="{TECHNICKA,STUDENTSKY_DUM}",
    )
    parser.add_argument("number_of_images", help="The number of images to scrape.", type=int)
    parser.add_argument("sleep_time", help="The time to wait between requests in seconds.", type=float, default=0.1)
    args = parser.parse_args()


    logger.info(f"Scraping {args.number_of_images} images from {args.menza.value}.")

    number_of_requests = args.number_of_images


    for i in range(number_of_requests):
        response = requests.get(Menza.get_menza_url(args.menza), timeout=15)
        response.raise_for_status()

        if response.status_code != 200:
            logger.warning(
                "Camera %s returned status code %d; exitting.",
                args.menza.value,
                response.status_code,
            )
            break

        if is_offline_image(response.content):
            logger.warning(
                "Camera %s is offline; exitting.",
                args.menza.value,
            )
            break

        current_time = int(time.time())
        with open(f'data/{args.menza.value}_{current_time}.jpg', 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded image {i+1}/{number_of_requests} at {current_time}")
        time.sleep(args.sleep_time)
            



if __name__ == "__main__":
    main()