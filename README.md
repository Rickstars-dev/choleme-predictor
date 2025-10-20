# Cholesterol Level Prediction Web App

A web-based regression application that predicts cholesterol levels based on health and lifestyle factors using NHANES government data.

## 🎯 Project Type
**Regression Model** - Predicts continuous cholesterol values (mg/dL)

## 📊 Dataset
**Source:** NHANES (National Health and Nutrition Examination Survey) - CDC Government Data
- Official US government health survey
- Cleaned and processed for machine learning
- Thousands of participants with comprehensive health data

## 🤖 Machine Learning Models
- **Random Forest Regressor** (Primary model - Best performance)
- **Linear Regression** (Baseline model)
- **Ridge Regression** (Regularized linear model)
- **Gradient Boosting Regressor** (Advanced ensemble method)

## 📈 Performance Metrics
Unlike classification (accuracy %), regression uses:
- **R² Score** - Percentage of variance explained (0-1, higher is better)
- **MAE (Mean Absolute Error)** - Average prediction error in mg/dL
- **RMSE (Root Mean Squared Error)** - Weighted error metric
- **MAPE** - Mean Absolute Percentage Error

## 🏥 Features (Input Variables)
- Age
- Gender
- BMI (Body Mass Index)
- Blood Pressure (Systolic/Diastolic)
- Physical Activity Level
- Smoking Status
- Diabetes Status
- Diet Quality Score
- Alcohol Consumption
- Other health indicators

## 🎯 Output
**Predicted Total Cholesterol Level** (e.g., 195 mg/dL)

### Interpretation:
- **Desirable:** < 200 mg/dL
- **Borderline High:** 200-239 mg/dL
- **High:** ≥ 240 mg/dL

## 🚀 Tech Stack
- **Backend:** Flask (Python web framework)
- **ML Library:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Deployment:** Render/PythonAnywhere/Railway

## 📁 Project Structure
```
cholesterol_predictor/
├── app.py                      # Flask web application
├── train_model.py              # Model training script
├── data/
│   ├── download_dataset.py     # NHANES data download script
│   └── nhanes_cholesterol.csv  # Cleaned dataset
├── model/
│   ├── cholesterol_model.pkl   # Trained model
│   └── scaler.pkl              # Feature scaler
├── templates/
│   └── index.html              # Web interface
├── requirements.txt            # Python dependencies
└── README.md
```

## 🛠️ Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   ```bash
   python data/download_dataset.py
   ```

3. **Train the model:**
   ```bash
   python train_model.py
   ```

4. **Run the app:**
   ```bash
   python app.py
   ```

5. **Open browser:**
   ```
   http://localhost:5000
   ```

## 🌐 Deployment
Ready to deploy on:
- Render (Recommended)
- PythonAnywhere
- Railway
- Heroku

See `DEPLOYMENT.md` for detailed instructions.

## 📚 Documentation
- `BEGINNERS_GUIDE.md` - Detailed code explanations
- `DEPLOYMENT.md` - Hosting instructions

## 🎓 Educational Value
This project demonstrates:
- **Regression modeling** (vs classification)
- Government dataset utilization
- Real-world health predictions
- Web application development
- Model deployment

## ⚠️ Disclaimer
This tool is for educational purposes only. Not intended for medical diagnosis. Consult healthcare professionals for actual health assessments.

---

**Built with 💙 using official CDC NHANES data**
