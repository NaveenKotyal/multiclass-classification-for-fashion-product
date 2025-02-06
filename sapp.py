import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
from model import FashionModel  # Import your trained model

# Load model
@st.cache_resource
def load_model():
    model = FashionModel()  # Ensure this matches your model class
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Load metadata
styles_df = pd.read_csv("styles.csv")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Streamlit UI
st.title("Fashion Product Attribute Predictor")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert to readable labels (Assume multi-output classification)
    pred = torch.argmax(output, dim=1).item()
    
    # Display prediction
    st.write(f"Predicted Category: {styles_df.iloc[pred]['category']}")
    st.write(f"Predicted Color: {styles_df.iloc[pred]['color']}")
    st.write(f"Predicted Gender: {styles_df.iloc[pred]['gender']}")
    st.write(f"Predicted Season: {styles_df.iloc[pred]['season']}")
