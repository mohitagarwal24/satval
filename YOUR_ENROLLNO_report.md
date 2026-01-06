# Satellite Imagery-Based Property Valuation
## Multimodal Regression Pipeline for Real Estate Analytics

---

**Author:** [Your Name]  
**Enrollment Number:** [YOUR_ENROLLNO]  
**Date:** January 2026

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Introduction & Objectives](#2-introduction--objectives)
3. [Data Overview](#3-data-overview)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Methodology](#5-methodology)
6. [Model Architecture](#6-model-architecture)
7. [Results & Analysis](#7-results--analysis)
8. [Model Explainability](#8-model-explainability)
9. [Conclusions](#9-conclusions)

---

## 1. Executive Summary

This project develops a **Multimodal Regression Pipeline** that predicts property market value by combining traditional tabular features with satellite imagery. The key findings are:

| Model | Validation RMSE | Validation R² |
|-------|-----------------|---------------|
| XGBoost (Tabular Only) | 0.1666 | 89.75% |
| ResNet18 (Image Only) | 0.4535 | 25.46% |
| **Stacking Fusion** | **0.1601** | **90.92%** |

**Key Achievement:** The multimodal fusion model achieved a **3.9% improvement** in RMSE over the tabular-only baseline, demonstrating that satellite imagery provides valuable supplementary information for property valuation.

---

## 2. Introduction & Objectives

### 2.1 Problem Statement
Real estate valuation traditionally relies on tabular data such as square footage, number of rooms, and location coordinates. However, visual characteristics like neighborhood quality, proximity to green spaces, and overall "curb appeal" are difficult to capture numerically.

### 2.2 Objectives
- Build a multimodal regression model to predict property prices
- Programmatically acquire satellite imagery using Google Maps Static API
- Perform exploratory and geospatial analysis
- Engineer features using CNNs to extract visual embeddings
- Test and compare fusion architectures
- Ensure model explainability using Grad-CAM

---

## 3. Data Overview

### 3.1 Dataset Statistics

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| Samples | 16,209 | 5,404 |
| Original Features | 21 | 20 |
| Engineered Features | 34 | 33 |
| Images Downloaded | 16,110 | 5,404 |

### 3.2 Target Variable
- **Price Range:** $75,000 - $7,700,000
- **Median Price:** $450,000
- **Transformation:** Log-transformed for stable training (range: 11.23 - 15.86)

### 3.3 Key Features

| Feature | Description |
|---------|-------------|
| sqft_living | Total interior living space |
| grade | Construction quality (1-13) |
| bathrooms | Number of bathrooms |
| view | View rating (0-4) |
| waterfront | Binary waterfront indicator |
| condition | Maintenance condition (1-5) |

### 3.4 Satellite Image Acquisition
- **API Used:** Google Maps Static API with URL Signing
- **Total Images:** 21,613 (16,209 training + 5,404 test)
- **Image Size:** 400×400 pixels
- **Zoom Level:** 18 (building-level detail)
- **Map Type:** Satellite

---

## 4. Exploratory Data Analysis

### 4.1 Data Quality
- **Missing Values:** None across all 21 columns
- **Data Types:** Appropriate (integers, floats for numerical features)

### 4.2 Price Distribution
The price distribution is heavily right-skewed, necessitating log-transformation:

```
Original Price Statistics:
- Minimum: $75,000
- Maximum: $7,700,000
- Median: $450,000

After Log Transform:
- Minimum: 11.23
- Maximum: 15.86
```

### 4.3 Feature Correlations with Price

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | sqft_living | 0.701 |
| 2 | grade | 0.664 |
| 3 | sqft_above | 0.603 |
| 4 | sqft_living15 | 0.582 |
| 5 | bathrooms | 0.525 |
| 6 | is_luxury* | 0.487 |
| 7 | quality_score* | 0.484 |
| 8 | total_rooms* | 0.464 |
| 9 | view | 0.391 |
| 10 | sqft_basement | 0.316 |

*Engineered features

### 4.4 Geospatial Analysis

**Waterfront Premium Analysis:**
- Non-waterfront average price: ~$450,000
- Waterfront average price: ~$1,350,000
- **Premium: 200.3%**

This significant premium highlights the importance of location-based features that satellite imagery can help capture.

### 4.5 Feature Engineering

13 new features were engineered from the original data:

| Feature | Description | Formula |
|---------|-------------|---------|
| is_renovated | Binary renovation flag | yr_renovated > 0 |
| house_age | Age of property | 2025 - yr_built |
| years_since_renovation | Years since last work | 2025 - yr_renovated_filled |
| total_rooms | Combined room count | bedrooms + bathrooms |
| bath_bed_ratio | Bathroom ratio | bathrooms / (bedrooms + 1) |
| living_lot_ratio | Space efficiency | sqft_living / sqft_lot |
| above_living_ratio | Above-ground ratio | sqft_above / sqft_living |
| has_basement | Basement indicator | sqft_basement > 0 |
| living_vs_neighbors | Relative size | sqft_living / sqft_living15 |
| lot_vs_neighbors | Relative lot size | sqft_lot / sqft_lot15 |
| quality_score | Overall quality | grade × condition |
| is_luxury | Luxury indicator | grade≥11 OR waterfront OR view≥3 |

---

## 5. Methodology

### 5.1 Overall Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                          │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │ Tabular Data │    │ Satellite Images (Google Maps)   │  │
│  │  (Excel)     │    │ 21,613 images @ 400×400px        │  │
│  └──────┬───────┘    └──────────────┬───────────────────┘  │
└─────────┼───────────────────────────┼──────────────────────┘
          │                           │
          ▼                           ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│   PREPROCESSING     │    │        IMAGE PIPELINE           │
│ • Feature Eng (13)  │    │ • Resize to 224×224             │
│ • Log Transform     │    │ • Normalize (ImageNet)          │
│ • Standard Scaling  │    │ • Data Augmentation             │
└──────────┬──────────┘    └──────────────┬──────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│     XGBoost         │    │        ResNet18                 │
│  (Tabular Model)    │    │     (Image Model)               │
│  R² = 89.75%        │    │     R² = 25.46%                 │
└──────────┬──────────┘    └──────────────┬──────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
              ┌───────────────────────┐
              │   STACKING FUSION     │
              │   (Ridge Regression)  │
              │   R² = 90.92%         │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  FINAL PREDICTIONS    │
              │  23323023_final.csv   │
              └───────────────────────┘
```

### 5.2 Data Split Strategy
- **Training:** 70% (11,346 samples)
- **Validation:** 30% (4,863 samples)
- **Random State:** 42 (for reproducibility)

---

## 6. Model Architecture

### 6.1 Baseline Model: XGBoost (Gradient Boosting)

**Hyperparameters:**
- n_estimators: 200
- learning_rate: 0.1
- max_depth: 5
- random_state: 42

**Input:** 31 scaled tabular features

### 6.2 Image Model: ResNet18

**Architecture:**
```
ResNet18 (Pretrained on ImageNet)
    │
    ├── Conv1 + BN + ReLU + MaxPool [FROZEN]
    ├── Layer1 (2 BasicBlocks) [FROZEN]
    ├── Layer2 (2 BasicBlocks) [TRAINABLE]
    ├── Layer3 (2 BasicBlocks) [TRAINABLE]
    ├── Layer4 (2 BasicBlocks) [TRAINABLE]
    ├── AdaptiveAvgPool2d
    │
    └── Regression Head:
        ├── Flatten
        ├── Linear(512 → 256) + ReLU + Dropout(0.5)
        └── Linear(256 → 1)
```

**Training Configuration:**
- Total Parameters: 11,308,097
- Trainable Parameters: 11,150,593
- Optimizer: Adam with differential learning rates
  - Backbone: 1e-5
  - Head: 1e-3
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Epochs: 15
- Batch Size: 32

### 6.3 Fusion Model: Stacking Meta-Learner

**Architecture:**
- Input: [XGBoost_prediction, CNN_prediction]
- Model: Ridge Regression (α = 1.0)
- Output: Final price prediction

**Learned Weights:**
- XGBoost contribution: ~95%
- CNN contribution: ~5%

---

## 7. Results & Analysis

### 7.1 Model Performance Comparison

| Model | Val RMSE | Val R² | Val MAE |
|-------|----------|--------|---------|
| XGBoost (Tabular) | 0.1666 | 89.75% | 0.1218 |
| ResNet18 (Image) | 0.4535 | 25.46% | 0.3512 |
| **Stacking Fusion** | **0.1601** | **90.92%** | **0.1171** |

### 7.2 Key Findings

**1. Tabular Features Dominate**
- XGBoost alone achieves R² = 89.75%
- This is expected because key price drivers (sqft_living, bedrooms, grade) are directly available in tabular form

**2. Image Model Adds Supplementary Value**
- CNN alone: R² = 25.46%
- The CNN captures visual patterns (lot size, neighborhood density, waterfront proximity)
- These patterns are visible from satellite but hard to quantify numerically

**3. Fusion Improves Performance**
- **3.9% RMSE reduction** compared to tabular-only
- Final R² = 90.92%
- The meta-learner learns optimal weighting: XGBoost dominates but CNN provides marginal improvements

### 7.3 CNN Training History

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1 | 4.3765 | 0.5704 |
| 5 | 1.7372 | 0.2829 |
| 10 | 1.5040 | 0.6489 |
| 14 | 1.3264 | **0.2052** |
| 15 | 1.3196 | 0.2305 |

**Best Model:** Epoch 14 (Val Loss = 0.2052)

### 7.4 Test Set Predictions

| Metric | Value |
|--------|-------|
| Mean Predicted Price | $539,464 |
| Minimum | $134,004 |
| Maximum | $5,645,792 |
| Total Predictions | 5,404 |

---

## 8. Model Explainability

### 8.1 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to understand what regions of satellite images the CNN focuses on for price prediction.

**Observations from Grad-CAM:**
1. **High-value properties:** Model focuses on waterfront areas, large lot boundaries
2. **Mid-range properties:** Attention on building footprint and surrounding structures
3. **Lower-value properties:** Focus on density of surrounding buildings, road proximity

### 8.2 Feature Importance (XGBoost)

| Feature | Importance |
|---------|------------|
| sqft_living | High |
| grade | High |
| lat/long | Medium |
| bathrooms | Medium |
| quality_score | Medium |

### 8.3 Interpretation

The CNN primarily captures:
- **Lot size and boundaries** (visible from satellite)
- **Building density** (neighborhood characteristic)
- **Proximity to water bodies** (waterfront detection)
- **Green space coverage** (environmental quality)

These visual features complement the tabular data, explaining the 3.9% improvement in the fusion model.

---

## 9. Conclusions

### 9.1 Summary

This project successfully developed a multimodal property valuation system that:

✅ Acquired 21,613 satellite images using Google Maps API  
✅ Engineered 13 new features from tabular data  
✅ Trained XGBoost (R² = 89.75%) and ResNet18 (R² = 25.46%) models  
✅ Achieved **90.92% R²** with stacking fusion  
✅ Demonstrated **3.9% RMSE improvement** over tabular-only baseline  
✅ Provided model explainability through Grad-CAM  

### 9.2 Why Image-Only Performance is Limited

Satellite images cannot capture:
- Interior features (sqft_living, bedrooms, bathrooms)
- Construction quality (grade)
- Maintenance condition
- Renovation status

These are the strongest price predictors, explaining why tabular data dominates.

### 9.3 Value of Multimodal Approach

Despite limitations, the CNN adds value by capturing:
- Visual neighborhood quality
- Lot boundaries and sizes
- Proximity to amenities (water, parks)
- Building density and urban context

### 9.4 Future Improvements

1. **Higher resolution imagery** (zoom level 19-20)
2. **Street-level images** (Google Street View) for curb appeal
3. **Temporal imagery** to detect seasonal patterns
4. **Advanced fusion** (attention mechanisms, cross-modal transformers)
5. **More training data** for better CNN generalization

---

## Appendix

### A. Technology Stack
- **Deep Learning:** PyTorch 2.x
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Image Processing:** OpenCV, PIL
- **API:** Google Maps Static API

### B. Repository Structure
```
satval/
├── data_fetcher.py          # Satellite image acquisition
├── preprocessing.ipynb       # Data cleaning and EDA
├── model_training.ipynb      # Model training and evaluation
├── YOUR_ENROLLNO_final.csv   # Test predictions
├── README.md                 # Project documentation
└── models/
    ├── cnn_model.pth
    ├── xgb_model.pkl
    ├── meta_learner.pkl
    └── gradcam_samples.png
```

### C. References
1. He, K., et al. "Deep Residual Learning for Image Recognition" (ResNet)
2. Selvaraju, R., et al. "Grad-CAM: Visual Explanations from Deep Networks"
3. Google Maps Static API Documentation

---

**End of Report**

