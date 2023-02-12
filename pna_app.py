import streamlit as st
import numpy as np
#import pandas as pd 
import pickle
#import requests
from PIL import Image
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array 
# import tensorflow.compat.v2 as tf


#Load model
#binary_model = pickle.load(open('base_binary_model.pkl','rb'))

#Load images
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img

#Predict Image
# def predict(img_file):
#     image = load_img(img_file,target_size=(256,256))
#     image = img_to_array(image)/255.0
#     image = image.reshape(1,256,256,3)
#     yhat = binary_model.predict(image)
#     label = np.where(yhat>0.5,"Pneumonia","Normal")
#     classification = f"The CXR suggests {label} with a prediction of {yhat}"
#     return classification
    
#Display screen
def main():
    st.title('Pneumonia Classifier')
    st.text('Provide a Chest X-Ray (CXR) image to predict if you have pneumonia (PNA).')
    #choose classifier
    model_selection = ['binary - (Normal, PNA)','multi-class - (Normal, Bacterial PNA, Viral PNA)']
    radio_selection = st.radio('Classifier Type',model_selection)
    #load image
    img_file = st.file_uploader("Upload Image", type=["jpeg","jpg","png"])
    if img_file is not None:
        st.image(load_image(img_file),width=255)
        classify = st.button('Classify Image')
        if classify:
            st.write("Classifying...")
            if radio_selection == 'binary - (Normal, PNA)':
                #load binary model
                st.write('predict with binary model')
            else:
                #load multiclass model
                st.write('predict with multi-class model')
            #predict image
            #st.write the result
                
        
if __name__ == '__main__':
    main()



