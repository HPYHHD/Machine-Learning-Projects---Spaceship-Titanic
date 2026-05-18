# Machine-Learning-Projects---Spaceship-Titanic
## Our group members and README of each:
## 1. Zhang Junhong: Baseline Model (Logistic Regression)


## 2. Zhao Yang: XGBoost Model ()


## 3. Gao Jie: LightGBM Model
## Spaceship Titanic - LightGBM Solution

This repository contains a complete machine learning pipeline for the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) Kaggle competition.  
The solution implements **super feature engineering**, **hyperparameter optimisation** (stochastic search), **threshold tuning**, **outlier engineering** and a final **LightGBM** model.

---

## 📁 Project Structure
├── train.csv # Training data (required)\
├── test.csv # Test data (required)\
├── space3.ipynb # Main notebook with all code\
├── submission_lightgbm_v2.csv # Output submission file\
├── requirements.txt # Environment & Dependencies\
└── README.md # This file

---

## 🐍 Environment & Dependencies

### Recommended setup
Create a fresh Python environment (Python 3.8+).  
All required packages are listed in `requirements.txt` below.

### Install dependencies

```bash
pip install -r requirements.txt

```

---

## 🚀 How to Run
1. Prepare data
Place train.csv and test.csv in the same directory as the notebook, or update the file paths in the notebook:

```bash
train_df = pd.read_csv(r'C:\Users\gj153\Desktop\train.csv')   # change this
test_df  = pd.read_csv(r'C:\Users\gj153\Desktop\test.csv')    # change this
```
2. 



