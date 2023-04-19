import cv2
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--video', default='./driving.mp4')
parser.add_argument('--output_dir', default='./video_output')

opt = parser.parse_args()

video_loc = opt.video
output_dir = opt.output_dir

video = cv2.VideoCapture(video_loc)

Path(output_dir).mkdir(exist_ok=True)

success,image = video.read()
count = 0
while success:
  cv2.imwrite(output_dir + "/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = video.read()
  count += 1
