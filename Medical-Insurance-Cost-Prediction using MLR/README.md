# 🏥 Medical Insurance Cost Prediction

## 🔍 Overview

This project predicts medical insurance charges using regression analysis based on demographic and health-related factors.

## 🎯 Objective

To identify key variables affecting insurance costs and build a predictive model.

## 📁 Dataset

* Features:

  * Age
  * BMI
  * Children
  * Smoker
  * Region
* Target Variable:

  * Insurance Charges

## 🧹 Data Preprocessing

* Outliers detected in BMI and treated
* Data distribution analyzed
* Feature selection performed

## 📊 Exploratory Data Analysis

* Strong correlation found:

  * Smoker vs Charges (high positive impact) 
* BMI also shows moderate impact on charges

## 🤖 Model Used

* Multiple Linear Regression (OLS)

## 📈 Model Performance

* R² Score: **0.740**
* Adjusted R²: **0.738**
* Model is statistically significant

## 🔑 Key Insights

* Major cost drivers:

  * Smoking status (strongest factor)
  * BMI
  * Age
* Insignificant variable removed:

  * Sex

## 📊 Statistical Validation

* No autocorrelation (Durbin-Watson ≈ 2.02)
* Residuals follow normal distribution
* Homoscedasticity satisfied

## 📁 Files Included

* Dataset (CSV)
* Python analysis code
* Final report (PDF)

## 📫 Author

Rahul Bhujade
