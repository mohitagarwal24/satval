# Satellite Imagery-Based Property Valuation

A multimodal deep learning system that predicts property prices by combining tabular data with satellite imagery.

## Overview

This project implements a **Multimodal Regression Pipeline** that predicts property market value using:
- **Tabular Data**: Property features (bedrooms, bathrooms, sqft, grade, etc.)
- **Satellite Imagery**: Visual environmental context captured via Google Maps API

The system uses a **stacking ensemble** approach:
1. **XGBoost**: Processes tabular features
2. **ResNet18**: Extracts visual features from satellite images
3. **Meta-Learner**: Combines both predictions for final output

## Project Structure

```
satval/
├── data_fetcher.py          # Satellite image download script
├── preprocessing.ipynb      # Data cleaning and EDA
├── model_training.ipynb     # Multimodal model training
├── code_with_output.ipynb   # Complete notebook containing code with outputs
├── README.md                # This file
│
|
|(Below directories will be added once you run the code_with_output.ipynb file)
├── data/
│   ├── raw/                 # Original Excel files
│   ├── processed/           # Processed CSV files
│   └── images/              # Downloaded satellite images
│       ├── train/
│       └── test/
│
└── models/                  # Saved models and visualizations
    ├── cnn_model.pth
    ├── xgb_model.pkl
    ├── meta_learner.pkl
    └── gradcam_samples.png
```
## Setup Guide (Kaggle) - Option- A

Directly go to below link and run the notebook. I have already downloaded the images which exists in /kaggle/working directory, so it will take less time to run. Make sure the accelerator is selected to GPU T4 X2.

Notebook - https://www.kaggle.com/code/mohitagarwal24/cdc-satellite-imagery-based-property-valuation

If you wish not to download images then I have made data uploaded to kaggle as well which can be accessed below for direct use, but these must be copied to /kaggle/working directory and should look like as follows:
<img width="279" height="328" alt="image" src="https://github.com/user-attachments/assets/b4450a84-b319-40be-9ed6-bdeea53a888c" />

Dataset - https://www.kaggle.com/datasets/mohitagarwal24/cdc-satellite-image-based-property-valuation-data

## Setup Guide (Kaggle) - Option- B

### 1. Create Kaggle Notebook
- Go to [kaggle.com/code](https://www.kaggle.com/code)
- Create new notebook with **GPU T4 x2**

### 2. Upload Data
- Download [train.xlsx](https://1drv.ms/x/c/8cf6803adf7941c3/IQBue1q4w4TETL_7xWMGhcD_AejALtdsXTBejVUjRA9qeM8?e=kWdglC) and [test.xlsx](https://1drv.ms/x/c/8cf6803adf7941c3/IQAwCVfSggmjQ4DJH51zJK-tARwRQWE9fl0bPlwo1mRF2PQ?e=h3frFB)
- Upload to Kaggle or add as dataset

### 3. Initial Setup
```python
import os
from pathlib import Path

# Create directories
for d in ['data/raw', 'data/processed', 'data/images/train', 'data/images/test', 'models']:
    Path(d).mkdir(parents=True, exist_ok=True)

# Copy data (after uploading to Kaggle)
!cp /kaggle/input/*/*.xlsx data/raw/

# Set your API key
os.environ['GOOGLE_MAPS_API_KEY'] = 'YOUR_GOOGLE_MAPS_API_KEY'
os.environ['GOOGLE_MAPS_SIGNING_SECRET'] = 'YOUR_SIGNING_SECRET'
```

### 4. Run Notebooks
1. Run `preprocessing.ipynb` - Data cleaning & EDA
2. Run `model_training.ipynb` - Train models & generate predictions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Data                              │
├─────────────────────────────┬───────────────────────────────┤
│     Satellite Image         │       Tabular Features         │
│       (224×224)             │       (31 features)            │
└──────────────┬──────────────┴───────────────┬───────────────┘
               │                               │
               ▼                               ▼
        ┌──────────────┐              ┌──────────────┐
        │   ResNet18   │              │   XGBoost    │
        │    (CNN)     │              │  (GBM 200)   │
        └──────┬───────┘              └──────┬───────┘
               │                               │
               │ Predictions                   │ Predictions
               │                               │
               └───────────┬───────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Meta-Learner  │
                  │    (Ridge)     │
                  └────────┬───────┘
                           │
                           ▼
                    Price Prediction
```

## Results

| Model | Val RMSE | Val R² |
|-------|----------|--------|
| XGBoost (Tabular) | ~0.166 | ~0.897 |
| ResNet18 (Image) | ~0.351 | ~0.552 |
| **Stacking (Fusion)** | **~0.159** | **~0.910** |


## Model Explainability

Grad-CAM is used to visualize which image regions influence predictions:
- Green areas (vegetation) → Higher value
- Water proximity → Higher value
- Urban density patterns → Variable impact

## Dataset Features

**Original Features (18)**
- Property: bedrooms, bathrooms, sqft_living, floors
- Quality: grade (1-13), condition (1-5), view (0-4)
- Location: lat, long, waterfront
- Temporal: yr_built, yr_renovated

**Engineered Features (13)**
- house_age, years_since_renovation
- total_rooms, bath_bed_ratio
- living_lot_ratio, quality_score
- is_luxury, has_basement, and more

## API Setup

### Google Maps Static API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project → Enable "Maps Static API"
3. Create API key → Enable billing


## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0
opencv-python>=4.8.0
tqdm>=4.65.0
openpyxl>=3.1.0
```

