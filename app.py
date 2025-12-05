import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
model = load_model('D:\PyCharm 2025.2.4\PythonProject\Image Classification 2\vgg19_model.keras')
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'radish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'tomato',
 'watermelon']

st.header('Image Classification')
img_height= 180
img_width= 180
image = st.text_input('Enter Image name ', 'chilli.jpg')
image_load = tf.keras.utils.load_img(image, target_size=(img_width, img_height))
img_arr = tf.keras.utils.img_to_array(image_load)
img_bat = tf.expand_dims(img_arr, axis=0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
#image = image[31:len(image)-1]
st.image(image)
st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
st.write(f'With accuracy of', np.max(score)*100)
