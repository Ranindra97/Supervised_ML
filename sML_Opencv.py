#!/user/bin/python3
import numpy as np
import cv2
from matplotlib import pyplot as plt
#loading/reading the image
#                (image name,image features)
img = cv2.imread('Dog.jpeg',1) #here 1 is by default for original image,we can use cv2.IMREAD_COLOR instead of 1
img1 = cv2.imread('Dog1.jpeg',0) #here 0 means no color,we can use cv2.IMREAD_BGR2GRAY instead of 0
img2 = cv2.imread('Dog1.jpeg',-1)  # here -1 for manage the tranceparency,we can use cv2.IMREAD_UNCHANGE_COLOR instead of 1

#print height and width

print(img.shape)
print(img1.shape)
print(img2.shape)

#To save the image which has makes some changes

cv2.imwrite("Black&White.jpeg",img1)

#displaying the image
#         (name by which you want to display the image,image data)
cv2.imshow("Dog",img)
cv2.imshow("Black&White",img1)
cv2.imshow("Tranceparent",img2)

#image window holder activate

cv2.waitKey(0)

#waitkey will distroy by using q button

cv2.destroyAllWindows()
