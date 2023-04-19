import cv2
import os
img=[]
dir = "/content/drive/MyDrive/Rob 498/Deep-WaveNet-Underwater-Image-Restoration/foggy/facades/Ours_EUVP/"
for i in os.listdir(dir):
    img.append(cv2.imread(dir+i))

height,width,layers=img[1].shape
output_dir=dir = "/content/drive/MyDrive/Rob 498/Deep-WaveNet-Underwater-Image-Restoration/foggy/"
video=cv2.VideoWriter(dir+'video.avi',cv2.VideoWriter_fourcc(*'MJPG'),30,(width,height))
for i in range(len(img)):
    video.write(img[i])

cv2.destroyAllWindows()
video.release()
