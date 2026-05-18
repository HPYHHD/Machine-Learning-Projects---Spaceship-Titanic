# Machine-Learning-Projects---Spaceship-Titanic
## Our group members and contributions:
## 1. Zhang Junhong: Baseline Model (Logistic Regression)

## 2. Zhao Yang: XGBoost Model

## 3. Gao Jie: LightGBM Model

## 4. Guo Sibo: Ensemble Model (XGBoost + LightGBM + CatBoost)

## 5. Gao Yuan: ANN (Neural Network)

## Spaceship Titanic - How to follow our projects

This repository contains a complete machine learning pipeline for the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) Kaggle competition.  
---

## 📁 Project Structure
├── train.csv # Training data (required)\
├── test.csv # Test data (required)\
├── Gao Jie - LightGBM Model.ipynb # LightGBM Model\
├── submission_lightgbm_v2.csv # Output of LightGBM Model\
├── Guo Sibo - Ensemble Model.py # Output of Ensemble Model\
├── submission_pure_weighted_thr46.csv # Output of Ensemble Model\
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
2. Run the notebook
Open space3.ipynb with Jupyter Notebook / JupyterLab / VS Code and execute all cells in order.

3. Output

Submission file: submission_lightgbm_v2.csv

Columns: PassengerId , Transported (True/False)

Feature importance plot: feature_importance_v2.png



