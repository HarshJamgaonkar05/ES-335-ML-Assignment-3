"""
Model file helper for Streamlit Cloud deployment
Downloads model files from external storage if not present locally
"""

import os
import streamlit as st

def ensure_models_exist():
    """
    Check if model files exist, provide instructions if missing.
    For Streamlit Cloud deployment, models need to be:
    1. Stored in GitHub LFS (Large File Storage), OR
    2. Downloaded from external hosting (Google Drive, Dropbox, etc.)
    """
    
    models_dir = "Models"
    required_models = [
        "shakespeare_best_model.pth",
        "linux_kernel_best_model.pth"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        st.error("### Model Files Missing")
        st.markdown("""
        The trained model files are not found. For deployment on Streamlit Cloud, 
        you have two options:
        
        #### Option 1: Use GitHub LFS (Recommended)
        ```bash
        # Install Git LFS
        git lfs install
        
        # Track large files
        git lfs track "Models/*.pth"
        
        # Add and commit
        git add .gitattributes Models/*.pth
        git commit -m "Add models with LFS"
        git push
        ```
        
        #### Option 2: External Hosting
        Upload your models to:
        - Google Drive (get shareable link)
        - Hugging Face Hub
        - AWS S3
        - Dropbox
        
        Then modify `app.py` to download them on startup.
        
        #### Option 3: Local Testing Only
        If running locally, ensure model files are in the `Models/` directory.
        
        **Missing files:**
        """)
        for model in missing_models:
            st.markdown(f"- `Models/{model}`")
        
        return False
    
    return True

# Example: Download from Google Drive (uncomment and customize)
"""
def download_from_gdrive(file_id, destination):
    import gdown
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

# Model file IDs (replace with your own)
MODEL_FILES = {
    'shakespeare_best_model.pth': 'YOUR_GDRIVE_FILE_ID_HERE',
    'linux_kernel_best_model.pth': 'YOUR_GDRIVE_FILE_ID_HERE'
}

def ensure_models_downloaded():
    os.makedirs('Models', exist_ok=True)
    for filename, file_id in MODEL_FILES.items():
        filepath = os.path.join('Models', filename)
        if not os.path.exists(filepath):
            st.info(f'Downloading {filename}...')
            download_from_gdrive(file_id, filepath)
    return True
"""
