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

def selected_layers_model(layer_names, baseline_model):
    outputs = [baseline_model.get_layer(name).output for name in layer_names]
    model = Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)

class StyleModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleModel, self).__init__()
        self.vgg = selected_layers_model(style_layers + content_layers, vgg)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

def image_to_style(image_tensor):
    extractor = StyleModel(style_layers, content_layers)
    return extractor(image_tensor)['style']

def style_to_vec(style):
    return np.hstack([np.ravel(s) for s in style.values()])

image_paths = glob.glob('/Users/tanayparikh/Desktop/Assignment-4/Fashion_images/*.jpg')

images = {}
image_style_embeddings = {}

for image_path in image_paths:
    image = cv2.imread(image_path, 3)
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    image = cv2.resize(image, (200, 200))
    images[ntpath.basename(image_path)] = image

for image_path in image_paths:
    image_tensor = load_image(image_path)
    style = style_to_vec(image_to_style(image_tensor))
    image_style_embeddings[ntpath.basename(image_path)] = style

tsne = manifold.TSNE(n_components=2, init='pca', perplexity=10, random_state=0)
X_tsne = tsne.fit_transform(np.array(list(image_style_embeddings.values())))

def embedding_plot(X, images, thumbnail_sparsity=0.005, thumbnail_size=0.3):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    fig, ax = plt.subplots(1, figsize=(12, 12))
    shown_images = np.array([[1., 1.]])
    for i in range(X.shape[0]):
        if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < thumbnail_sparsity:
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        thumbnail = offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r, zoom=thumbnail_size)
        ax.add_artist(offsetbox.AnnotationBbox(thumbnail, X[i], bboxprops=dict(edgecolor='white'), pad=0.0))
    plt.grid(True)


# Create a Streamlit app
st.title("Image Similarity Search: Developing a Content-Based Image Retrieval System")

# Upload a reference image
user_image = st.file_uploader("Upload your reference image", type=["jpg", "jpeg"])

if user_image is not None:
    user_image = load_image(user_image)
    user_style_embedding = style_to_vec(image_to_style(user_image))

    # Perform style-based image search
    if st.button("Search for Images by Style"):
        st.write("Searching for images by style...")
        st.write("Please wait, this might take a moment.")


        def search_by_style(image_style_embeddings, images, reference_image, max_results=3):
            v0 = reference_image
            distances = {}
            for k, v in image_style_embeddings.items():
                d = sc.spatial.distance.cosine(v0, v)
                distances[k] = d

            sorted_neighbors = sorted(distances.items(), key=lambda x: x[1], reverse=False)

            f, ax = plt.subplots(1, max_results, figsize=(16, 8))
            for i, img in enumerate(sorted_neighbors[:max_results]):
                ax[i].imshow(images[img[0]])
                ax[i].set_axis_off()

            st.pyplot(f)

        search_by_style(image_style_embeddings, images, user_style_embedding)



