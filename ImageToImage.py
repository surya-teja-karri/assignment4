import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import cv2
import numpy as np
import ntpath
from sklearn.metrics.pairwise import cosine_similarity
import scipy as sc
import glob
import matplotlib.pyplot as plt
from sklearn import manifold
from matplotlib import offsetbox

# Load your code for style embeddings and searching

def load_image(image):
    image = plt.imread(image)
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf.image.resize(img, [400, 400])
    img = img[tf.newaxis, :]
    return img

content_layers = ['block5_conv2']

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    # 'block4_conv1',
    # 'block5_conv1'
]



