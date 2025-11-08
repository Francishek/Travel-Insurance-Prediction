# DS.v3.3.1.5

# Travel Insurance Prediction Data Project

## Introduction

**Context**

In 2019, a Tour & Travels company launched a new travel insurance package that included coverage for COVID-related risks. The company is now seeking to identify which customers are most likely to purchase this insurance based on historical data collected during the initial offering. This dataset includes information on nearly 2,000 previous customers, capturing a variety of demographic, financial, and behavioral features.

**Goals**

The objective of this project is to develop a predictive model that can intelligently determine whether a customer would be interested in buying the travel insurance package. 

By understanding customer profiles and their influence on insurance purchasing decisions, the company aims to optimize its marketing strategy and offer personalized promotions — ultimately improving sales efficiency and customer targeting.

The outcome of this project may support customer-specific advertising and has the potential to help families make informed travel decisions — possibly saving money on unexpected travel-related expenses.

**Dataset Overview**

The dataset contains 1987, 9 records of previous customers who were offered travel insurance in 2019. It includes **8 independent variables**:

Age- Age Of The Customer

Employment Type- The Sector In Which Customer Is Employed

GraduateOrNot- Whether The Customer Is College Graduate Or Not

AnnualIncome- The Yearly Income Of The Customer In Indian Rupees[Rounded To Nearest 50 Thousand Rupees]

FamilyMembers- Number Of Members In Customer's Family

ChronicDisease- Whether The Customer Suffers From Any Major Disease Or Conditions Like Diabetes/High BP or Asthama,etc.

FrequentFlyer- Derived Data Based On Customer's History Of Booking Air Tickets On Atleast 4 Different Instances In The Last 2 Years[2017-2019].

EverTravelledAbroad- Has The Customer Ever Travelled To A Foreign Country[Not Necessarily Using The Company's Services]

and **target variable**:

TravelInsurance- Did The Customer Buy Travel Insurance Package During Introductory Offering Held In The Year 2019.

### Requirements for Jupyter Notebook:
- Python 3.11.9
- Pandas 2.2.3
- NumPy 1.26.4
- Seaborn 0.13.2
- Matplotlib 3.10.1
- Statsmodels 0.14.4
- Scipy 1.11.4
- Sklearn: 1.6.1

## Data Source

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data).

Jupyter Notebook and dataset source, clone the Repository:
[GitHub](https://github.com/TuringCollegeSubmissions/fverko-DS.v2.5.3.1.5)

## Jupyter Notebook Structure:

## 1. Introduction

## 2. Exploratory Data Analysis (EDA)

### A. Data loading & Initial checks

### B. Univariate Analysis

### C. Multivariate Analysis

## 3. Statistical Inference

## 4. Machine Learning Modeling

### A. Data Preparation

### B. Pipeline Preprocessing

### C. Model Selection

### D. Hyperparameter Tuning 

### E. Ensembling

### F. Models evaluation on Test set

### G. Feature Importance Analysis

## 5. Conclusion

This project focused on predicting whether customers would purchase travel insurance using a range of machine learning models. After extensive preprocessing, model tuning, and evaluation, a tuned  ensemble model (Random Forest and Hist Gradient Boostings) achieved the best performance. Random forest was best in single models. Feature importance analysis revealed that variables like Annual Income, Age, Family members, and Frequent Flyer status played a key role in prediction. The models were further validated using cross-validation, ROC curves, confusion matrices, and decision boundary visualizations.






















