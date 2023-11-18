import streamlit as st
import clip
import torch
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import os

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to the images
images_path = "/Users/tanayparikh/Desktop/Assignment-4/Fashion_images"

# Load features
features_path = "./features"
image_ids = pd.read_csv(Path(features_path) / 'image_ids.csv')
image_ids = list(image_ids['image_id'])
image_features = np.load(Path(features_path) / 'features.npy')


# Convert features to Tensors
if device == "cpu":
    image_features = torch.from_numpy(image_features).float().to(device)
else:
    image_features = torch.from_numpy(image_features).to(device)

# Function to encode search query
def encode_search_query(search_query):
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded

# Function to find best matches
def find_best_matches(text_features, image_features, image_ids, results_count=6):
    similarities = (image_features @ text_features.T).squeeze(1)
    best_image_idx = (-similarities).argsort()
    return [image_ids[i] for i in best_image_idx[:results_count]]

# Function to perform search
def search(search_query, image_features, image_ids, results_count=3):
    text_features = encode_search_query(search_query)
    return find_best_matches(text_features, image_features, image_ids, results_count)

# Streamlit app
st.title("CLIP-based Image Search App")

# Sidebar with search query input
search_query = st.sidebar.text_input("Enter a feature:")
if search_query:
    results = search(search_query, image_features, image_ids)
    st.subheader("Search Results:")
    
    # Display images in a row
    col1, col2, col3 = st.columns(3)
    for result in results:
        image = Image.open(os.path.join(images_path, f"{result}.jpg"))
        col1.image(image, caption=result, use_column_width=True)

    # Optionally, display additional information or details about the images

# You can add more Streamlit components as needed.

