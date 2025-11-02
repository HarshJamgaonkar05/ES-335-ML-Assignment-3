# 🚀 Streamlit Cloud Deployment Guide

## Problem: Model Files Too Large

Your model files (~700 MB total) are too large for standard GitHub repositories. Here are **3 solutions**:

---

## ✅ Solution 1: Git LFS (Recommended for Streamlit Cloud)

### Step 1: Install Git LFS
```bash
# Download and install from: https://git-lfs.github.com/
# Or via package manager:
# Windows (Chocolatey): choco install git-lfs
# Mac (Homebrew): brew install git-lfs
# Linux (apt): sudo apt install git-lfs

git lfs install
```

### Step 2: Track Large Files
```bash
cd h:\College\Sem5\ML\ES-335-ML-Assignment-3

# Track all .pth files
git lfs track "Models/*.pth"

# This creates/updates .gitattributes
git add .gitattributes
```

### Step 3: Add Models to LFS
```bash
# Add model files
git add Models/*.pth

# Commit
git commit -m "Add model files with Git LFS"

# Push (this will upload to LFS)
git push
```

### Step 4: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set main file: `app.py`
4. Deploy!

**Note**: Streamlit Cloud supports Git LFS out of the box.

---

## ✅ Solution 2: Google Drive + Auto-Download

### Step 1: Upload Models to Google Drive
1. Upload both `.pth` files to Google Drive
2. Right-click → Share → Get shareable link
3. Extract file ID from URL:
   - URL: `https://drive.google.com/file/d/1abc123xyz/view`
   - File ID: `1abc123xyz`

### Step 2: Add gdown to requirements.txt
```bash
echo "gdown>=4.7.1" >> requirements.txt
```

### Step 3: Modify app.py
Add this at the top of `app.py` (before model loading):

```python
import gdown
import os

def download_models_if_needed():
    """Download models from Google Drive if not present"""
    MODEL_IDS = {
        'shakespeare_best_model.pth': 'YOUR_SHAKESPEARE_FILE_ID',
        'linux_kernel_best_model.pth': 'YOUR_LINUX_FILE_ID'
    }
    
    os.makedirs('Models', exist_ok=True)
    
    for filename, file_id in MODEL_IDS.items():
        filepath = f'Models/{filename}'
        if not os.path.exists(filepath):
            st.info(f'⏳ Downloading {filename}... (first time only)')
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filepath, quiet=False)
            st.success(f'✅ Downloaded {filename}')

# Call before loading models
download_models_if_needed()
```

### Step 4: Update .gitignore
```bash
echo "Models/*.pth" >> .gitignore
```

### Step 5: Deploy
1. Commit and push changes (WITHOUT model files)
2. Deploy on Streamlit Cloud
3. App will download models on first run

**Pros**: No large files in repo  
**Cons**: Slower first load (one-time download)

---

## ✅ Solution 3: Hugging Face Hub

### Step 1: Upload to Hugging Face
```bash
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Upload models
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='Models/shakespeare_best_model.pth',
    path_in_repo='shakespeare_best_model.pth',
    repo_id='YOUR_USERNAME/next-word-mlp',
    repo_type='model'
)
"
```

### Step 2: Modify app.py
```python
from huggingface_hub import hf_hub_download

def download_from_hf(filename):
    """Download model from Hugging Face Hub"""
    return hf_hub_download(
        repo_id="YOUR_USERNAME/next-word-mlp",
        filename=filename,
        cache_dir="./Models"
    )

# Use in load_model function
model_path = download_from_hf('shakespeare_best_model.pth')
```

---

## 🔧 Quick Fix for Current Deployment Issue

The immediate problem is **not** the model files, but the **old pickle5 dependency**. 

### Fix Applied:
Updated `requirements.txt` to remove pickle5 (not needed in Python 3.8+)

### Redeploy Steps:
1. **Commit the updated requirements.txt**:
   ```bash
   git add requirements.txt .streamlit/
   git commit -m "Fix: Remove pickle5 dependency for Python 3.13 compatibility"
   git push
   ```

2. **On Streamlit Cloud**:
   - Click "Reboot app" or redeploy
   - Watch logs for successful install

---

## 📊 Current Requirements (Fixed)

```
numpy>=1.21.0
torch>=2.0.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
streamlit>=1.28.0
```

**Removed**: `pickle5` (built into Python 3.8+)  
**Removed**: Heavy packages not needed for inference (jupyter, notebook, torchvision, seaborn)

---

## 🐛 Troubleshooting

### Issue: "Models not found"
- **Local**: Ensure `Models/` directory has the `.pth` files
- **Cloud**: Use Git LFS or auto-download solution

### Issue: "Out of memory"
- PyTorch models are large; Streamlit Cloud has 1GB RAM limit
- Consider model quantization or smaller architecture

### Issue: "Slow startup"
- Model loading takes time (normal for first load)
- Use `@st.cache_resource` (already implemented in app.py)

### Issue: "Requirements install failed"
- Check Python version compatibility
- Remove version pins if conflicts occur
- Use `--prefer-binary` flag

---

## 📝 Recommended Deployment Checklist

- [ ] Remove `pickle5` from requirements.txt ✅ (DONE)
- [ ] Create `.streamlit/config.toml` ✅ (DONE)
- [ ] Add `.gitattributes` for LFS ✅ (DONE)
- [ ] Choose model hosting solution (LFS/Drive/HF)
- [ ] Test locally: `streamlit run app.py`
- [ ] Commit all changes
- [ ] Push to GitHub
- [ ] Deploy on Streamlit Cloud
- [ ] Monitor logs for errors

---

## 🎯 Recommended: Git LFS

For your use case, **Git LFS is the best solution** because:
1. ✅ Streamlit Cloud supports it natively
2. ✅ No code changes needed
3. ✅ Fast deployment
4. ✅ Version control for models

### Quick Setup:
```bash
# Install LFS
git lfs install

# Track models
git lfs track "Models/*.pth"

# Add everything
git add .gitattributes Models/*.pth .streamlit/ requirements.txt

# Commit
git commit -m "Setup Git LFS for model files"

# Push
git push
```

---

## 🆘 Need Help?

If deployment still fails, check:
1. **Streamlit Cloud Logs**: Look for specific error messages
2. **Model File Sizes**: Confirm they're in LFS (GitHub shows "Stored with Git LFS")
3. **Python Version**: Streamlit Cloud uses Python 3.9-3.11 by default
4. **RAM Usage**: Large models may exceed free tier limits

---

**Your app is ready to deploy once you choose a model hosting solution!** 🚀
