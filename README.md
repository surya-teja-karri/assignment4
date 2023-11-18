Assignment4

Overview
This project implements an advanced image search application leveraging the CLIP model for embedding calculation and Streamlit for user interaction. It allows users to search for images from a fashion dataset based on text queries and find similar images to a given image. The project involves key steps like computing and storing image embeddings, setting up cloud storage and a database, and developing a web interface using Streamlit and FAST API.

Prerequisites
Python 3.8 or later
PyTorch
CLIP (Contrastive Language-Image Pretraining) model
Pinecone for embedding storage
Amazon S3 for image storage
FAST API for backend development
Streamlit for frontend development

Step 1: Compute and Store Image Embeddings
Load CLIP Model: Use the CLIP model from OpenAI to generate embeddings for images in the fashion dataset.
Process Images: Preprocess the images using CLIP's preprocessing tools.
Compute Embeddings: Compute the embedding for each image with the CLIP model.
Store Embeddings: Save these embeddings in Pinecone for efficient similarity searches.

Step 2: Set Up Cloud Storage and Database
Upload Images to Amazon S3: Store the processed images in an Amazon S3 bucket.
Database Management: Develop a database linking image IDs to text tags and embeddings.

Step 3: Develop Backend with FAST API
API Function for Text-based Search: Create a function in FAST API to retrieve the closest image matching a text description by querying Pinecone with text query embeddings.
API Function for Similarity Search: Implement an additional function to find images similar to an uploaded image or an image from a URL. This function will compare the embedding of the input image against the database to find the closest matches.
Setup and Deployment: Establish the FAST API framework and deploy the application.

Step 4: Create a Streamlit Web Interface
User Interface Design: Use Streamlit to design an interactive interface for users to input text queries and upload or link images.
Integration with FAST API: Ensure the Streamlit app can communicate with the FAST API backend to fetch search results based on user text input or image uploads.
Deployment: Deploy the Streamlit application.

Running the Application
Launch the FAST API server.
Start the Streamlit app.
Input a text query or upload/link an image in the Streamlit interface to receive relevant image results or similar images from the fashion dataset.
