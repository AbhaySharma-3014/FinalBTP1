import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import imutils
from displayTumor import *


st.title("Brain Tumor Detection")
st.write("By Abhay | Akash | Aman | Himanshu")

st.write("NSUT BTP Project-2022")
uploaded_file = st.file_uploader("Choose a image file", type="jpg")



# *******************************************************Models************************************************************#
prediction_model = tf.keras.models.load_model("./Detection/detection_model.hdf5")
classification_model = tf.keras.models.load_model("./Classification/classification_model.hdf5")
# *************************************************************************************************************************#





# ************************************************Decoding Image Data************************************************************#
if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

# *************************************************************************************************************************#



#1. PreProcessing   

    img = cv2.resize(opencv_image,(224,224))
    
    # **********************GrayScaling****************************
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # ********************Filtering (Gaussian Filter) **************
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    
        
#2. Segmentation     
    # ********************Thresholding Segmentation*****************
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # ********************Watershed Segmentation********************
        
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # Finding sure background area
    sure_bg = cv2.dilate(thresh,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1


    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    im1 = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    
# ************************************************************************************************************************




# ****************************************************Display**************************************************************#   
    st.image(im1, channels="RGB")
# *************************************************************************************************************************#
    




# *****************************************************Button**************************************************************#
    Genrate_pred = st.button("Generate Prediction")    
# *************************************************************************************************************************#
    




# *****************************************************Prediction***********************************************************#    
    if Genrate_pred:       
        
        res = prediction_model.predict(image)
        
        if(res < 0):
            st.title("No Tumor is Present")
        else:
            
            #prediction resulted from classification_model
            prediction = classification_model.predict(uploaded_file).argmax()

            #category resulted from classification_model            
            class_dict = {0: "glioma_tumor", 1: "meningioma_tumor", 2: "pituitary_tumor"}
            curr_cat=class_dict [prediction]
            
            st.title("Tumor is Present and category is {}".format(curr_cat))

# *************************************************************************************************************************#





            
#********************************************Things for showing region of tumor********************************************#
    Img = uploaded_file
    curImg=uploaded_file

    gray = cv2.cvtColor(np.array(Img), cv2.COLOR_BGR2GRAY)
    [ret, thresh] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    curImg = opening

    # Finding sure background area
    sure_bg = cv2.dilate(curImg, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(curImg, cv2.DIST_L2, 5)
    [ret, sure_fg] = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region (Subtracting)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg,dtype=cv2.CV_32F)


    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now mark the region of unknown with zero
    markers[unknown == 255] = 0   
    markers = cv2.watershed(opencv_image, markers)

    opencv_image[markers == -1] = [255, 0, 0]

    tumorImage = cv2.cvtColor(opencv_image, cv2.COLOR_HSV2BGR)
    curImg = tumorImage            
    st.image(curImg)
# *************************************************************************************************************************#