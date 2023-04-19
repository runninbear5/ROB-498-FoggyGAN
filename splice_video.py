import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', default='./driving.mp4')
parser.add_argument('--output_dir', default='./video_output')

video_loc = parser.video
output_dir = parser.output_dir

video = cv2.VideoCapture(video_loc)

success,image = video.read()
count = 0
while success:
  cv2.imwrite(output_dir + "/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = video.read()
  count += 1
