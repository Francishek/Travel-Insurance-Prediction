# Travel Insurance Prediction Data Project

## Introduction

**Context**

In 2019, a Tour & Travels company launched a new travel insurance package that included coverage for COVID-related risks. The company is now seeking to identify which customers are most likely to purchase this insurance based on historical data collected during the initial offering. This dataset includes information on nearly 2,000 previous customers, capturing a variety of demographic, financial, and behavioral features.

**Goals**

    To buil d a predictive model that accurately identifies customers who are likely to purchase a travel insurance package, enabling improved customer targeting and more effective marketing strategies.

**Objectives**

    Understand Customer Profiles: Analyze customer demographic and travel-related attributes to identify patterns associated with travel insurance purchasing behavior.
    
    Prepare and Model Data: Clean, preprocess, and engineer relevant features from the Kaggle travel insurance dataset to develop a classification model.
    
    Model Evaluation: Train and compare multiple machine learning algorithms to determine the best-performing model based on accuracy and other relevant evaluation metrics.
    
    Provide Actionable Insights: Translate model findings into meaningful insights that can guide targeted marketing strategies and personalized promotional offers.
    
    Scope Limitation: The model will use only the attributes available in the provided dataset and will focus on prediction and insights, not real-time deployment or pricing optimization.

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

## Data Source

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data).

## Setup

1. Clone the repo:

   Link [GitHub](https://github.com/Francishek/Travel-Insurance-Prediction) or
   ```bash
   git clone https://github.com/Francishek/Travel-Insurance-Prediction
   cd project-root
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

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






















