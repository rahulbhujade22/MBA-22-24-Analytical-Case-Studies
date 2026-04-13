# 🚗 MPG Prediction using Linear Regression

## 🔍 Overview

This project uses multiple linear regression to predict car fuel efficiency (MPG).

## 📁 Dataset

* Features include:

  * Displacement
  * Horsepower
  * Weight
  * Acceleration
  * Model Year
  * Origin

## 📊 Exploratory Data Analysis

* Distribution analysis for all variables
* Statistical summary computed
* Outliers detected using boxplots

## 🧹 Data Preprocessing

* Outliers handled:

  * MPG: removed 1 outlier
  * Horsepower: replaced with whisker value
  * Acceleration: removed ~10 outliers
* Multicollinearity removed using correlation analysis:

  * Removed displacement, weight, cylinders

## 🤖 Model Used

* Multiple Linear Regression → Final model simplified

## 📈 Final Model

Only one significant variable remained:

**Price = 3.2587 + 0.01556 × Horsepower**

## 📊 Model Evaluation

* R² Score: **0.0149** (low explanatory power)
* Model statistically significant (p < 0.05)

## 🔑 Insights

* Horsepower is the only significant predictor
* Other variables showed multicollinearity
* Model has low predictive strength

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib
jupyter notebook
```

## 📷 Output

* Histograms
* Boxplots
* Correlation matrix
