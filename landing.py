import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import sys
import os
from PIL import Image

# Set page configuration ONCE at the very beginning
st.set_page_config(
    page_title="Glaucoma Detection Hub",
    page_icon="👁️",
    layout="wide"
)

# Create a session state object if it doesn't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    st.button("🏠 Home", on_click=navigate_to, args=('home',), use_container_width=True)
    st.button("🔄 ResNet-18", on_click=navigate_to, args=('resnet',), use_container_width=True)
    st.button("🎯 YOLO & XGBoost", on_click=navigate_to, args=('yolo',), use_container_width=True)
    st.button("🔍 U-Net", on_click=navigate_to, args=('unet',), use_container_width=True)
    st.button("📊 Model Comparison", on_click=navigate_to, args=('comparison',), use_container_width=True)
    st.button("⚙️ Preprocessing", on_click=navigate_to, args=('preprocessing',), use_container_width=True)

def render_home():
    st.title("👁️ Glaucoma Detection Hub")
    st.write("Compare different deep learning models for Glaucoma detection")

    st.markdown("""
    ### Available Models
    Choose from our selection of advanced glaucoma detection models:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### ResNet-18 Model")
        st.write("Uses ResNet-18 architecture for binary classification")
        st.button("Launch ResNet-18 Model", on_click=navigate_to, args=('resnet',))

    with col2:
        st.info("### YOLO & XGBoost Model")
        st.write("Combines YOLO object detection with XGBoost classification")
        st.button("Launch YOLO & XGBoost Model", on_click=navigate_to, args=('yolo',))

    with col3:
        st.info("### U-Net Model")
        st.write("Utilizes U-Net architecture for segmentation")
        st.button("Launch U-Net Model", on_click=navigate_to, args=('unet',))

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

def render_model(model_name):
    st.title(f"{model_name} Model")
    st.write(f"This is the {model_name} model interface.")
    
    # Import and run the specific model code here
    try:
       if model_name == "ResNet-18":
           sys.path.append(os.path.join(os.getcwd(), "Resnet 18"))
           import resnet_glaucoma
           resnet_glaucoma.main()
       elif model_name == "YOLO & XGBoost":
           sys.path.append(os.path.join(os.getcwd(), "YOLO and XGBoost"))
           import yolov8_glaucoma_
           yolov8_glaucoma_.main()  # Call main function here
       elif model_name == "U-Net":
           sys.path.append(os.path.join(os.getcwd(), "U-Net"))
           import glaucocare
           glaucocare.main()
    except Exception as e:
        st.error(f"Error loading {model_name} model: {str(e)}")
        st.write("Please make sure all required files and dependencies are available.")

# Main content router
if st.session_state.current_page == 'home':
    render_home()
elif st.session_state.current_page == 'comparison':
    # Directly import and call the function from comparison.py
    try:
        sys.path.append(os.path.join(os.getcwd(), "comparison"))
        import comp
        comp.main()  # Assuming the main function renders comparison content
    except Exception as e:
        st.error(f"Error loading comparison page: {str(e)}")
elif st.session_state.current_page == 'resnet':
    render_model("ResNet-18")
elif st.session_state.current_page == 'yolo':
    render_model("YOLO & XGBoost")
elif st.session_state.current_page == 'unet':
    render_model("U-Net")
elif st.session_state.current_page == 'preprocessing':
    try:
        import preprocessing_glaucoma
        preprocessing_glaucoma.main()
    except Exception as e:
        st.error(f"Error loading preprocessing tool: {str(e)}")
        st.write("Please make sure all required files and dependencies are available.")

# Add a home button in the footer for easy navigation
if st.session_state.current_page != 'home':
    st.markdown("---")
    st.button("🏠 Return to Home", on_click=navigate_to, args=('home',))
