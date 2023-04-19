import cv2

video = cv2.VideoCapture("/content/drive/MyDrive/Rob 498/Deep-WaveNet-Underwater-Image-Restoration/foggy/driving.mp4")

success,image = video.read()
count = 0
while success:
  cv2.imwrite("/content/drive/MyDrive/Rob 498/Deep-WaveNet-Underwater-Image-Restoration/foggy/real_input/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = video.read()
  print('Read a new frame: ', success)
  count += 1
