#importing
import streamlit as st
import numpy as np
from tensorflow import keras 
import cv2 
import os

import subprocess
if not os.path.isfile('model1.h5'):
    subprocess.run(['curl --output model1.h5 "https://media.githubusercontent.com/media/lokkenchan/PNA_Classifier/main/Binary_RN50_TF_NO_ES_031823.h5"'], shell=True)

#Load models and compile
binary_model = keras.models.load_model('model1.h5', compile=False)

binary_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','AUC','Recall'])

multiclass_model = keras.models.load_model('Multiclass_RN50_TF_NO_ES_031823.h5',compile=False)
multiclass_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','AUC','Recall'])

#Focus On Lungs
def focus_on_lungs(img):
    #Turn the image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to smooth the binary mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Perform connected component analysis to find the largest connected component
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    mask = np.uint8(labels == largest_label)

    # Find the bounding box of the largest connected component
    x, y, w, h = cv2.boundingRect(mask)

    # Crop the original image using the bounding box
    cropped = img[y:y+h, x:x+w]
    return cropped

#Normalize and Reshape before model prediction
def normalize_reshaped(img):
    img_normalized = img/255.0
    width = 224
    height = 224
    dim = (width,height)
    normalized_reshaped = cv2.resize(img_normalized,dim,interpolation=cv2.INTER_AREA)
    return normalized_reshaped

#Predict Image with Binary Model
#input after preprocessing the image is a numpy ndarray with a shape of 224,224,3 and dtype offloat64
def binary_predict(img_array):
    img_array = np.reshape(img_array,(1,224,224,3))
    yhat = binary_model.predict(img_array)
    label = np.where(yhat>0.5,"Pneumonia","Normal")
    if label == "Normal":
        yhat= 1-yhat
    classification = f"The CXR suggests {label[0][0]} with a prediction of {yhat[0][0]}"
    return classification

#Predict Image with Multiclass Model
def multiclass_predict(img_array):
    img_array = np.reshape(img_array,(1,224,224,3))
    softmax_prediction = multiclass_model.predict(img_array)
    predictions=np.zeros_like(softmax_prediction)
    predictions[np.arange(len(softmax_prediction)),np.argmax(softmax_prediction,axis=1)]=1
    idx=np.argmax(predictions)
    if idx == 0:
        label = 'Bacterial PNA'
    elif idx == 1:
        label = 'Normal'
    else:
        label = 'Viral PNA'
    classification = f"The CXR suggests {label} with a prediction of {np.max(softmax_prediction[0])}"
    return classification

    
#Display screen
def main():
    #Description
    st.title('Pneumonia Classifier')
    st.text('Provide a jpg/jpeg Chest X-Ray (CXR) image to predict if you have pneumonia (PNA).')
    st.text('The binary & multiclass ResNet-50 models have a 90 & 85% test accuracy respectively.') 
    st.text('WARNING: This application is LIMITED & is NOT a substitute for medical advice.')
    
    #Choose Classifier
    model_selection = ['binary - (Normal, PNA)','multi-class - (Bacterial PNA, Normal, Viral PNA)']
    radio_selection = st.radio('Classifier Type',model_selection)

    #Load Image
    uploaded_file = st.file_uploader("Upload Image", type=["jpeg","jpg"])
    if uploaded_file is not None:
        #Convert the uploaded file into a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
        #Decode numpy array to image using opencv
        img = cv2.imdecode(file_bytes,1)
        #Display image
        st.image(img)
        classify = st.button('Classify Image')
        #If button is pressed
        if classify:
            st.write("Classifying...")
            #preprocess
            img = focus_on_lungs(img)
            img = normalize_reshaped(img)
            #Classifier selection conditional
            if radio_selection == 'binary - (Normal, PNA)':
                #predict with binary classifier
                st.write(binary_predict(img))
            else:
                #predict with multiclass classifier
                st.write(multiclass_predict(img))
                
        
if __name__ == '__main__':
    main()



