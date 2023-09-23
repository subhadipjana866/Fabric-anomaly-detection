import numpy as np
import cv2
import streamlit as st
from PIL import Image


def defect_detect(image):
    img = image.copy()
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    blr = cv2.blur(v,(15,15))
    dst = cv2.fastNlMeansDenoising(blr,None,10,7,21)
    _,binary = cv2.threshold(dst,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.erode(binary,kernel,iterations=1)
    dilation = cv2.dilate(binary,kernel,iterations=1)

    if(dilation == 0).sum() > 1:
        id = "Defective Fabric"
        contours,_ = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in contours:
            if cv2.contourArea(i) < 261121.0:
                cv2.drawContours(img,i,-1,(255,0,0),3)
    else:
        id = "Good Fabric"
    return img,id

image = st.file_uploader("Upload the fabric image",type=['jpg','png'])
if image is not None:

    input_image = Image.open(image)
    st.image(input_image,width=300,caption="Original Image")  
    image = np.array(input_image.convert('RGB'))
    outimage,id = defect_detect(image)
    st.image(outimage,width=300,caption=id)

    
