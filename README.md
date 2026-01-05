# Satellite Imagery-Based Property Valuation

A multimodal deep learning system that predicts property prices by combining tabular data with satellite imagery.

**Reference**: Based on [D3vutkarsh/satellite-property-valuation](https://github.com/D3vutkarsh/satellite-property-valuation)

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
├── README.md                # This file
│
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

## Quick Start (Kaggle)

### 1. Create Kaggle Notebook
- Go to [kaggle.com/code](https://www.kaggle.com/code)
- Create new notebook with **GPU T4 x2**

### 2. Upload Data
- Download [train.xlsx](https://1drv.ms/x/c/8cf6803adf7941c3/IQBue1q4w4TETL_7xWMGhcD_AejALtdsXTBejVUjRA9qeM8?e=kWdglC) and [test.xlsx](https://1drv.ms/x/c/8cf6803adf7941c3/IQAwCVfSggmjQ4DJH51zJK-tARwRQWE9fl0bPlwo1mRF2PQ?e=h3frFB)
- Upload to Kaggle or add as dataset

### 3. Set API Key
```python
import os
os.environ['GOOGLE_MAPS_API_KEY'] = 'your_api_key_here'
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
| XGBoost (Tabular) | ~0.18 | ~0.88 |
| ResNet18 (Image) | ~0.25 | - |
| **Stacking (Fusion)** | **~0.16** | **~0.90** |

*RMSE in log space*

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

## Submission Files

1. **Prediction CSV**: `enrollno_final.csv`
   - Format: `id, predicted_price`

2. **Report PDF**: `enrollno_report.pdf`
   - EDA visualizations
   - Architecture diagram
   - Model comparison results
   - Grad-CAM examples

3. **Code Repository**:
   - data_fetcher.py
   - preprocessing.ipynb
   - model_training.ipynb
   - README.md

## API Setup

### Google Maps Static API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project → Enable "Maps Static API"
3. Create API key → Enable billing
4. Cost: ~$50 (covered by $200/month free credit)

```python
os.environ['GOOGLE_MAPS_API_KEY'] = 'your_key'
```

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

## License

MIT License

## Acknowledgments

- Dataset: King County House Sales
- Reference: [D3vutkarsh/satellite-property-valuation](https://github.com/D3vutkarsh/satellite-property-valuation)
- Satellite Imagery: Google Maps Static API

