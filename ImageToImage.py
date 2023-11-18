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





