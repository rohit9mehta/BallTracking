from matplotlib import pyplot as plt
from cv2 import cv2
import numpy as np
import os
import re

#listing down all the file names
frames = os.listdir('frames/')
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

#reading frames
images=[]
for i in frames:
    img = cv2.imread('frames/'+i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(25,25),0)
    images.append(img)

images=np.array(images)

nonzero=[]

for i in range((len(images)-1)):
    mask = cv2.absdiff(images[i],images[i+1])
    _ , mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    num = np.count_nonzero((mask.ravel()))
    nonzero.append(num)
    
    
x = np.arange(0,len(images)-1)
y = nonzero

# plt.figure(figsize=(20,4))
# plt.scatter(x,y)
# plt.show()

#establish threshold for when frame becomes too "different"
#now frames with pitch in view are isolated
threshold = 15 * 10e3
for i in range(len(images)-1):
    if(nonzero[i]>threshold): 
        scene_change_idx = i
        break
        
frames = frames[:(scene_change_idx+1)]


img= cv2.imread('frames/' + frames[10])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(25,25),0)

# plt.figure(figsize=(5,10))
# plt.imshow(gray,cmap='gray')
# plt.show()

#white is 255, black is 0. any object below 200 marked as 0, any >200 marked as 255
_ , mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# plt.figure(figsize=(5,5))
# plt.imshow(mask,cmap='gray')
# plt.show()

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_copy = np.copy(gray)
cv2.drawContours(img_copy, contours, -1, (0,255,0), 3)
plt.imshow(img_copy, cmap='gray')
plt.show()