import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./video_output')
parser.add_argument('--output_video', default='./results.avi')

dir = parser.input_dir
output_video = parser.output_video

img=[]
for i in os.listdir(dir):
    img.append(cv2.imread(dir+i))

height,width,layers=img[1].shape
video=cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc(*'MJPG'),30,(width,height))
for i in range(len(img)):
    video.write(img[i])

cv2.destroyAllWindows()
video.release()
