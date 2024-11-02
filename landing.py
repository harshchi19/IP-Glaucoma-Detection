import streamlit as st
import sys
import os
from PIL import Image
import subprocess
import webbrowser

def main():
    st.set_page_config(
        page_title="Glaucoma Detection Hub",
        page_icon="👁️",
        layout="wide"
    )

    # Header
    st.title("👁️ Glaucoma Detection Hub")
    st.write("Compare different deep learning models for Glaucoma detection")

    # Main content area
    st.markdown("""
    ### Available Models
    Choose from our selection of advanced glaucoma detection models:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### ResNet-18 Model")
        st.write("Uses ResNet-18 architecture for binary classification")
        if st.button("Launch ResNet-18 Model"):
            subprocess.Popen(["streamlit", "run", 
                            os.path.join("Resnet 18", "glaucoma.py")])

    with col2:
        st.info("### YOLO & XGBoost Model")
        st.write("Combines YOLO object detection with XGBoost classification")
        if st.button("Launch YOLO & XGBoost Model"):
            subprocess.Popen(["streamlit", "run", 
                            os.path.join("YOLO and XGBoost", "glaucoma_yolov8.py")])

    with col3:
        st.info("### U-Net Model")
        st.write("Utilizes U-Net architecture for segmentation")
        if st.button("Launch U-Net Model"):
            subprocess.Popen(["streamlit", "run", 
                            os.path.join("U-Net", "glaucocare.py")])

    # Model Comparison Section
    st.markdown("---")
    st.header("Model Comparison")
    if st.button("Launch Model Comparison"):
        subprocess.Popen(["streamlit", "run", "comparison.py"])

    # Preprocessing Section
    st.markdown("---")
    st.header("Image Preprocessing")
    if st.button("Launch Preprocessing Tool"):
        subprocess.Popen(["streamlit", "run", "Glaucoma-preprocessing.py"])

    # Footer with model information
    st.markdown("---")
    st.markdown("""
    ### Model Information
    
    1. **ResNet-18 Model**
    - Architecture: Deep Residual Network (18 layers)
    - Task: Binary Classification (Normal/Glaucoma)
    - Accuracy: 99.83%
    
    2. **YOLO & XGBoost Model**
    - Architecture: YOLO for segmentation + XGBoost for classification
    - Features: CDR, RDR, NRR metrics calculation
    - Multiple detection metrics
    
    3. **U-Net Model**
    - Architecture: U-Net for semantic segmentation
    - Specialized in optic disc segmentation
    """)

if __name__ == "__main__":
    main()