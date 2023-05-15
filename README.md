# PNA Classifier
## Classifying Chest X-rays for Presence of Pneumonia
**Background:**

Healthcare providers help save lives and service so many people. Within medicine there are many diagnostic tools, and for this project the focus will be on medical imaging. With so many X-rays being read, it would be beneficial to reduce their workload to be allocated elsewhere. This could mean patients can get more time with the provider, it could be used to service more people, or even to create a more sustainable work-force. The benefit of having a web application solution is that the use would extend to those around the world as long as they have an internet connection thus improving accessibilty.

As for the disease, Pneumonia is the inflammation of the alveoli (of the lungs) with the air space filling with pus and fluid. It can be caused by different pathogens, namely, bacteria and viruses. It also notably impacts children around the world. The dataset that I obtained was from [Guangzhou Women and Children's Medical Center](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) consisting of chest x-rays that are labeled as normal and pneumonia. Within the titles of the images, those that are pneumonia are named as either bacteria or viral in cause. Because of this, we have the potential to create a binary and multi-class classifier. A binary classifier can classify images as either normal or pneumonia. A multi-class classifier can classify the images among normal, bacterial pneumonia, and viral pneumonia.

![image](https://user-images.githubusercontent.com/12520975/228328584-228e974d-bc8f-44ae-a588-0633a936fcd4.png)

**How to use:**

1. Select a model, binary or multi-class, you would like to apply on the image.
2. Click the "Browse files" button to select a jpeg/jpg image to load.
3. The image will appear on the screen as well as a "Classify Image" button.
4. Click on the "Classify Image" button to receive a prediction of the chest x-ray (CXR) with its confidence level.

The confidence ranges from 0.0 to 1.0 meaning from 0% to 100% confident. 

*At the moment, this application serves purely as a proof of concept demonstrating model creation, tuning, evaluating, and deployment.*

**Link to application:**

https://lokkenchan-pna-classifier-pna-app-ygl1p7.streamlit.app/
