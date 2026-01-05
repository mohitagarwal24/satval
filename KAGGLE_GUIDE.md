# Kaggle Quick Start Guide

## Step 1: Create Kaggle Notebook (2 min)
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Settings → Accelerator → **GPU T4 x2**
4. Save

## Step 2: Upload Data (3 min)
1. Download data files:
   - [train.xlsx](https://1drv.ms/x/c/8cf6803adf7941c3/IQBue1q4w4TETL_7xWMGhcD_AejALtdsXTBejVUjRA9qeM8?e=kWdglC)
   - [test.xlsx](https://1drv.ms/x/c/8cf6803adf7941c3/IQAwCVfSggmjQ4DJH51zJK-tARwRQWE9fl0bPlwo1mRF2PQ?e=h3frFB)
2. In Kaggle: Add Data → Upload → Select files

## Step 3: Setup (Run First Cell)
```python
# Setup
import os
from pathlib import Path

# Create directories
for d in ['data/raw', 'data/processed', 'data/images/train', 'data/images/test', 'models']:
    Path(d).mkdir(parents=True, exist_ok=True)

# Copy data
!cp /kaggle/input/*/*.xlsx data/raw/

# Set API key (IMPORTANT: Replace with your key!)
os.environ['GOOGLE_MAPS_API_KEY'] = 'YOUR_API_KEY_HERE'

print("✓ Setup complete!")
```

## Step 4: Download Images (~40 min)
Run the image download cell from `data_fetcher.py`

## Step 5: Run Preprocessing
Copy cells from `preprocessing.ipynb`

## Step 6: Train Models (~45 min)
Copy cells from `model_training.ipynb`

## Step 7: Generate Predictions
Final cell creates `enrollno_final.csv`

---

## Time Estimates

| Step | Time |
|------|------|
| Setup | 5 min |
| Image Download | 40 min |
| Preprocessing | 3 min |
| XGBoost Training | 5 min |
| CNN Training | 30 min |
| Fusion | 5 min |
| Predictions | 2 min |
| **Total** | **~1.5 hours** |

---

## Submission Checklist

- [ ] `enrollno_final.csv` - Replace with your enrollment number
- [ ] `enrollno_report.pdf` - Create report with EDA, architecture, results
- [ ] Push code to GitHub
- [ ] Submit to [portal](https://forms.gle/aw1jewkBQGeKStH37)

**Deadline: 5 January 2026 (EOD)**

