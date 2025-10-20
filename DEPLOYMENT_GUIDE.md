# CholeME Deployment Guide

## Option 1: Deploy to Render (RECOMMENDED - FREE & EASY)

### Prerequisites
1. Create a free GitHub account (if you don't have one): https://github.com
2. Create a free Render account: https://render.com

### Step 1: Prepare Your Project

Your project is already set up! All required files are in place:
- ✅ `requirements.txt` (lists all Python packages)
- ✅ `app.py` (main application)
- ✅ `.gitignore` (prevents unnecessary files from uploading)

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `choleme-predictor`
3. Description: "CholeME - AI-powered cholesterol level predictor"
4. Make it Public (required for free tier)
5. Click "Create repository"

### Step 3: Upload Code to GitHub

Open PowerShell in your project folder and run:

```powershell
cd "C:\Users\Abhishek Chandra\Documents\Personal(ForRicky)\Projects\cholesterol_predictor"

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - CholeME cholesterol predictor"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/choleme-predictor.git

# Push to GitHub
git push -u origin main
```

If git asks for credentials, use your GitHub username and create a Personal Access Token:
- Go to: https://github.com/settings/tokens
- Generate new token (classic)
- Select "repo" scope
- Use the token as your password

### Step 4: Deploy on Render

1. Go to https://render.com and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub account
4. Select your `choleme-predictor` repository
5. Configure:
   - **Name**: `choleme-predictor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`

6. Click "Create Web Service"

### Step 5: Wait for Deployment

- Render will automatically build and deploy your app (takes 2-3 minutes)
- You'll get a URL like: `https://choleme-predictor.onrender.com`
- Your app is now live! 🎉

---

## Option 2: Deploy to PythonAnywhere (Alternative)

### Step 1: Create Account
1. Go to https://www.pythonanywhere.com/registration/register/beginner/
2. Choose a username (this will be in your URL)
3. Free tier is perfect for this app

### Step 2: Upload Files
1. Click "Files" tab
2. Create folder: `cholesterol_predictor`
3. Upload all your files:
   - `app.py`
   - `requirements.txt`
   - `templates/` folder
   - `model/` folder
   - `data/` folder

### Step 3: Setup Virtual Environment
Open a Bash console and run:
```bash
mkvirtualenv --python=/usr/bin/python3.10 choleme
pip install -r cholesterol_predictor/requirements.txt
```

### Step 4: Configure Web App
1. Go to "Web" tab
2. Click "Add a new web app"
3. Choose "Flask"
4. Python version: 3.10
5. Path to Flask app: `/home/YOUR_USERNAME/cholesterol_predictor/app.py`
6. Set virtualenv: `/home/YOUR_USERNAME/.virtualenvs/choleme`

### Step 5: Configure WSGI
Edit WSGI configuration file:
```python
import sys
path = '/home/YOUR_USERNAME/cholesterol_predictor'
if path not in sys.path:
    sys.path.append(path)

from app import app as application
```

7. Click "Reload" to start your app
8. Visit: `https://YOUR_USERNAME.pythonanywhere.com`

---

## Option 3: Railway (Fast Deployment)

### Step 1: Create Account
- Go to https://railway.app
- Sign in with GitHub

### Step 2: Deploy
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your `choleme-predictor` repo
4. Railway auto-detects Python and deploys!
5. Click "Generate Domain" to get your URL

---

## Important Notes

### File Size Limits
- Model files (`model/cholesterol_model.pkl`) might be large
- If deployment fails due to size:
  1. Compress the model
  2. Use Git LFS (Large File Storage)
  3. Or retrain with a simpler model

### Environment Variables (if needed)
Most platforms let you set environment variables:
- Go to Settings → Environment Variables
- Add any sensitive data there (API keys, etc.)

### Free Tier Limitations
- **Render**: App sleeps after 15 min of inactivity (wakes up automatically when visited)
- **PythonAnywhere**: Limited CPU, no always-on
- **Railway**: 500 hours/month free

---

## Which Should You Choose?

| Feature | Render | PythonAnywhere | Railway |
|---------|--------|----------------|---------|
| Ease | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | Fast | Medium | Very Fast |
| Free Tier | Good | Good | Good |
| Custom Domain | ✅ | ✅ (paid) | ✅ |
| HTTPS | ✅ | ✅ | ✅ |

**Recommendation: Start with Render** - easiest and most reliable!

---

## After Deployment

1. Test your live URL
2. Share it in your presentation
3. Update the creator name if needed
4. Monitor usage in the dashboard

Need help with any step? Let me know!
