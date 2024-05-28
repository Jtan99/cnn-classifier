# CNN classifier

**Table of Contents**

- [Introduction](#introduction)
- [Features Overview](#features-overview)
- [Requirements](#requirements)
- [Configuration and Setup Instructions](#configuration-and-setup-instructions)
- [(Optional) Customize the Model and App](#optional-customize-the-model-and-app)

## Introduction
Welcome to the CNN Classifier Project! This project utilizes TensorFlow to build a convolutional neural network (CNN) model capable of classifying images of various fruits and vegetables. Additionally, we have developed a simple web application using Streamlit, allowing users to upload their own images for classification.


## Features Overview
- **Image Classification**: The core feature of the project is the ability to classify images into different categories of fruits and vegetables.
- **User-Friendly Web Interface**: A Streamlit-based web application that allows users to easily upload images and view classification results.
- **Real-time Predictions**: Get instant classification results as soon as the image is uploaded.
- **Extensive Dataset**: The model is trained on a comprehensive dataset containing a wide variety of fruits and vegetables.


## Requirements
To run this project, you need to have the following software and libraries installed:

- Python 3.10.6

It is recommended to use a virtual environment to manage dependencies. You can install the required libraries listed in `requirements.txt`.

### Configuration and Setup Instructions  
1. **Create a virtual environment**:  
    ```
    python -m venv venv
    ```

2. **Navigate to the Project Directory**:  
    ```
    cd cnn-classifier
    ```

3. **Set Up the Virtual Environment**:  
Follow the instructions in the Requirements section to create and activate a virtual environment, and install the required dependencies.

4. **Download and Prepare Dataset**:  
Download the dataset (if not included in the repository) and place it in the appropriate directory. (TODO: include link here)

5. **Train the Model**:  
Open and run the Jupyter notebook classifier.ipynb to train the TensorFlow model. This notebook will load the dataset, preprocess the images, and train the model. The trained model will be saved as Image_classify.keras.

6. **Run the Streamlit Web Application**:  
Launch the Streamlit web app to start classifying images:  
    ```
    streamlit run app.py
    ```
    This will open a new tab in your web browser where you can upload images and view the classification results.

### Optional: Customize the Model and App

- **Model Customization**: Modify `classifier.ipynb` to experiment with different model architectures or hyperparameters.
- **App Customization**: Customize the Streamlit app by editing `app.py` to add more features or improve the user interface.
