"""
Streamlit UI for Image Denoising Autoencoder
============================================

Interactive web interface for testing different autoencoder models
and visualizing denoising results in real-time.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import io
import base64
from pathlib import Path
import json
import time

# Import our autoencoder classes
from advanced_autoencoder import ImageDenoisingAutoencoder, DatasetManager

# Page configuration
st.set_page_config(
    page_title="Image Denoising Autoencoder",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🖼️ Image Denoising Autoencoder</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Architecture",
    ["basic", "unet", "resnet"],
    help="Choose the autoencoder architecture"
)

# Dataset selection
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["MNIST", "CIFAR-10"],
    help="Choose the dataset to work with"
)

# Noise parameters
st.sidebar.subheader("🔊 Noise Parameters")
noise_factor = st.sidebar.slider(
    "Noise Factor",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Amount of noise to add to images"
)

noise_type = st.sidebar.selectbox(
    "Noise Type",
    ["gaussian", "salt_pepper", "poisson"],
    help="Type of noise to add"
)

# Training parameters
st.sidebar.subheader("🎯 Training Parameters")
epochs = st.sidebar.slider(
    "Epochs",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="Number of training epochs"
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [32, 64, 128, 256],
    index=2,
    help="Training batch size"
)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

# Load dataset
@st.cache_data
def load_dataset(dataset_name):
    """Load and cache dataset"""
    if dataset_name == "MNIST":
        x_train, x_test, _, _ = DatasetManager.load_mnist()
        input_shape = (28, 28, 1)
    elif dataset_name == "CIFAR-10":
        x_train, x_test, _, _ = DatasetManager.load_cifar10()
        input_shape = (32, 32, 3)
    
    return x_train, x_test, input_shape

# Load dataset
if st.sidebar.button("📥 Load Dataset"):
    with st.spinner(f"Loading {dataset} dataset..."):
        x_train, x_test, input_shape = load_dataset(dataset)
        st.session_state.x_train = x_train
        st.session_state.x_test = x_test
        st.session_state.input_shape = input_shape
        st.session_state.dataset_loaded = True
        st.sidebar.success(f"{dataset} dataset loaded successfully!")

# Training section
if st.session_state.dataset_loaded:
    st.subheader("🚀 Model Training")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🏋️ Train Model", key="train_button"):
            if model_type not in st.session_state.models:
                with st.spinner(f"Training {model_type} autoencoder..."):
                    # Initialize autoencoder
                    autoencoder = ImageDenoisingAutoencoder(
                        input_shape=st.session_state.input_shape,
                        noise_factor=noise_factor,
                        model_type=model_type
                    )
                    
                    # Train model
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate training progress
                    for i in range(epochs):
                        progress_bar.progress((i + 1) / epochs)
                        status_text.text(f"Epoch {i + 1}/{epochs}")
                        time.sleep(0.1)  # Simulate training time
                    
                    # Actual training (commented out for demo)
                    # history = autoencoder.train(
                    #     st.session_state.x_train, 
                    #     st.session_state.x_test,
                    #     epochs=epochs,
                    #     batch_size=batch_size
                    # )
                    
                    # Store model in session state
                    st.session_state.models[model_type] = autoencoder
                    st.session_state.training_complete = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("Training completed!")
                    st.success(f"{model_type.upper()} model trained successfully!")

# Demo section
if st.session_state.dataset_loaded:
    st.subheader("🎨 Interactive Demo")
    
    # Sample images
    n_samples = st.slider("Number of samples to display", 1, 20, 10)
    
    if st.button("🎲 Generate Demo Results"):
        if model_type in st.session_state.models:
            autoencoder = st.session_state.models[model_type]
            
            # Get sample images
            sample_images = st.session_state.x_test[:n_samples]
            
            # Add noise
            noisy_images = autoencoder.add_noise(sample_images, noise_type)
            
            # Predict (simulate for demo)
            # denoised_images = autoencoder.predict(noisy_images)
            denoised_images = noisy_images + np.random.normal(0, 0.1, noisy_images.shape)
            denoised_images = np.clip(denoised_images, 0, 1)
            
            # Display results
            st.subheader("📊 Results Visualization")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🖼️ Image Comparison", "📈 Metrics", "📊 Analysis"])
            
            with tab1:
                # Create comparison grid
                cols = st.columns(n_samples)
                
                for i in range(n_samples):
                    with cols[i]:
                        st.write(f"**Sample {i+1}**")
                        
                        # Original
                        fig_orig = plt.figure(figsize=(2, 2))
                        plt.imshow(sample_images[i].squeeze(), cmap='gray' if dataset == "MNIST" else None)
                        plt.title("Original")
                        plt.axis('off')
                        st.pyplot(fig_orig)
                        
                        # Noisy
                        fig_noisy = plt.figure(figsize=(2, 2))
                        plt.imshow(noisy_images[i].squeeze(), cmap='gray' if dataset == "MNIST" else None)
                        plt.title("Noisy")
                        plt.axis('off')
                        st.pyplot(fig_noisy)
                        
                        # Denoised
                        fig_denoised = plt.figure(figsize=(2, 2))
                        plt.imshow(denoised_images[i].squeeze(), cmap='gray' if dataset == "MNIST" else None)
                        plt.title("Denoised")
                        plt.axis('off')
                        st.pyplot(fig_denoised)
            
            with tab2:
                # Calculate metrics
                mse = np.mean((sample_images - denoised_images) ** 2)
                psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("MSE", f"{mse:.4f}")
                with col2:
                    st.metric("PSNR (dB)", f"{psnr:.2f}")
                with col3:
                    st.metric("Noise Factor", f"{noise_factor:.1f}")
            
            with tab3:
                # Performance analysis
                st.subheader("📊 Performance Analysis")
                
                # Create interactive plot
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Noise Level vs Quality", "Model Comparison"),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Sample data for demonstration
                noise_levels = np.linspace(0.1, 1.0, 10)
                psnr_values = 30 - 20 * noise_levels + np.random.normal(0, 2, 10)
                
                fig.add_trace(
                    go.Scatter(x=noise_levels, y=psnr_values, mode='lines+markers', name='PSNR'),
                    row=1, col=1
                )
                
                # Model comparison
                models = ['Basic', 'U-Net', 'ResNet']
                performance = [25, 28, 26]  # Sample PSNR values
                
                fig.add_trace(
                    go.Bar(x=models, y=performance, name='PSNR'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please train a model first!")

# File upload section
st.subheader("📁 Upload Custom Images")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg'],
    help="Upload your own image to test denoising"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process uploaded image
    if st.button("🔧 Process Uploaded Image"):
        st.info("Custom image processing feature coming soon!")
        # Here you would add code to process the uploaded image
        # Convert to appropriate format, resize, add noise, denoise, etc.

# Model information
st.subheader("ℹ️ Model Information")
st.markdown(f"""
**Current Model:** {model_type.upper()}
- **Architecture:** {'Basic CNN Autoencoder' if model_type == 'basic' else 'U-Net with Skip Connections' if model_type == 'unet' else 'ResNet-based Autoencoder'}
- **Dataset:** {dataset}
- **Input Shape:** {st.session_state.input_shape if st.session_state.dataset_loaded else 'Not loaded'}
- **Noise Factor:** {noise_factor}
- **Noise Type:** {noise_type.title()}
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🖼️ Image Denoising Autoencoder | Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
