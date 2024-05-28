import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
import streamlit as st

def read_csv_to_list(file_path):
  df = pd.read_csv(file_path, header=None)
  items = df[0].tolist()
  return items

def make_prediction(image):
  img_height = 180
  img_width = 180

  image = image.resize((img_height, img_width))
  img_arr = tf.keras.utils.img_to_array(image)
  img_tensor = tf.convert_to_tensor(img_arr, dtype=tf.float32)
  img_bat = tf.expand_dims(img_tensor, 0)
  prediction = model.predict(img_bat)
  return prediction

model_path = os.path.join(os.getcwd(),"Image_classify.keras")
model = load_model(model_path)
data_cat = read_csv_to_list("categories.csv")

# path = os.path.join(os.getcwd(),"Fruits_vegetables","test", "carrot", "Image_1.jpg")

# image = tf.keras.utils.load_img(path, target_size=(img_height, img_width))
# img_arr = tf.keras.utils.array_to_img(image)
# img_bat = tf.expand_dims(img_arr, 0)

# predict = model.predict(img_bat)

st.header('Image Classification Model')
st.write("Upload an of a image fruit/vegetable to classify")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  resized_image = image.resize((360, 360))
  st.image(resized_image, caption="Uploaded Image", use_column_width="auto")

  prediction = make_prediction(image)
  score = tf.nn.softmax(prediction)

  st.write('Image is a(n) {}'.format(data_cat[np.argmax(score)]))
  st.write('Accuracy of {:0.2f}%'.format(np.max(score)*100))