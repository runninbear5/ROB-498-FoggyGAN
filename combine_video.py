import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./video_results')
parser.add_argument('--output_video', default='./results.avi')

opt = parser.parse_args()

dir = opt.input_dir
output_video = opt.output_video

img=[]
img_names = os.listdir(dir)
img_names.sort()
for i in img_names:
    img.append(cv2.imread(dir+i))

height,width,layers=img[1].shape
video=cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc(*'MJPG'),30,(width,height))
for i in range(len(img)):
    video.write(img[i])

cv2.destroyAllWindows()
video.release()
