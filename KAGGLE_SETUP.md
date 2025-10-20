# 🔑 Kaggle API Setup Guide

## Step 1: Create Kaggle Account (if you don't have one)

1. Go to: **https://www.kaggle.com**
2. Click "Register" (top right)
3. Sign up with Google/Email
4. Verify your email

---

## Step 2: Get Your API Key

1. **Log in to Kaggle**
2. Click your **profile picture** (top right)
3. Select **"Settings"**
4. Scroll down to **"API"** section
5. Click **"Create New API Token"**
6. A file named `kaggle.json` will download automatically

**The file looks like:**
```json
{
  "username": "your_username",
  "key": "your_secret_key_here"
}
```

⚠️ **IMPORTANT:** Keep this file secure! It's like a password.

---

## Step 3: Place the API Key

**Windows:**

Create this folder:
```
C:\Users\Abhishek Chandra\.kaggle\
```

Move `kaggle.json` to:
```
C:\Users\Abhishek Chandra\.kaggle\kaggle.json
```

**I've already created the folder for you!**

---

## Step 4: Verify Setup

Once you've placed the file, I'll verify it works and download the dataset!

---

## Quick Checklist

- [ ] Kaggle account created
- [ ] API token downloaded (kaggle.json)
- [ ] File placed in: C:\Users\Abhishek Chandra\.kaggle\kaggle.json
- [ ] Ready to download data!

---

**After you complete these steps, let me know and I'll download the real NHANES data!** 🚀
