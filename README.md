# Diabetes Patient Readmission Prediction Using Machine Learning

This project investigates the factors contributing to the readmission of diabetes patients within 30 days of discharge. Using a healthcare dataset, we evaluated machine learning models to predict readmission while addressing class imbalance and extracting actionable insights to improve patient outcomes.


## Problem Statement

Hospital readmissions for diabetes patients are a significant healthcare challenge, contributing to increased costs and resource utilization. Traditional prediction methods fail to capture the complexity of healthcare data, leading to suboptimal interventions. Machine learning provides scalable, data-driven solutions to predict patient readmission and guide effective strategies for intervention.


## Objectives

- Develop interpretable machine learning models to predict 30-day readmissions for diabetes patients.
- Address the class imbalance issue inherent in readmission datasets using techniques like class weighting.
- Evaluate models using comprehensive metrics: accuracy, precision, recall, F1 score, and AUC-ROC.
- Provide actionable insights to guide healthcare decisions and reduce readmission rates.


## Research Questions

1. What are the primary predictors of patient readmissions within 30 days?  
2. How effective are class weighting techniques in addressing imbalanced healthcare datasets?  
3. Which machine learning models provide the best trade-off between predictive performance and interpretability for readmission prediction?


## Dataset

### Source
- **Platform**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)  
- **Size**: 100,748 rows and 49 attributes  

### Key Features

| **Feature Name**      | **Description**                             | **Type**         |
|------------------------|---------------------------------------------|------------------|
| `age`                 | Patient's age bracket                      | Categorical      |
| `gender`              | Gender of the patient                      | Categorical      |
| `race`                | Race of the patient                        | Categorical      |
| `time_in_hospital`    | Duration of hospital stay                  | Continuous       |
| `num_lab_procedures`  | Number of lab procedures performed          | Continuous       |
| `num_medications`     | Number of medications administered          | Continuous       |
| `number_outpatient`   | Number of outpatient visits                 | Continuous       |
| `number_emergency`    | Number of emergency visits                  | Continuous       |
| `number_inpatient`    | Number of inpatient admissions              | Continuous       |
| `readmitted`          | Target variable (readmitted within 30 days)| Categorical      |


## Tools and Libraries

- Python  
- **Libraries**:
  - Data Processing: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine Learning: `scikit-learn`, `tensorflow`  
  - Class Imbalance: `imbalanced-learn`  


## Project Workflow

### 1. Data Preprocessing
- Categorical encoding for variables like `race`, `gender`, and `age`.
- Addressed missing values by replacing them with appropriate measures (e.g., mode for categorical features).
- Applied scaling (`MaxAbsScaler`) to normalize numerical features.

### 2. Class Imbalance Handling
- Incorporated `class_weight='balanced'` in Random Forest to address imbalance.
- SMOTE oversampling was tested but removed for better generalization.

### 3. Modeling
- Evaluated three primary models:
  - Decision Tree
  - Random Forest with class weights
  - Neural Network
  - Convolutional Neural Network (CNN)

### 4. Evaluation Metrics
  - **Accuracy**-Overall correctness of predictions.  
  - **Precision**-Fraction of correctly predicted positives.  
  - **Recall**-Fraction of actual positives identified.  
  - **F1 Score**-Harmonic mean of precision and recall.  
  - **AUC-ROC**-Model’s ability to distinguish between classes.  


## Results

| Model                      | Accuracy | Precision | Recall | F1 Score | Support (Class 1) |
|----------------------------|----------|-----------|--------|----------|-------------------|
| **Decision Tree**          | 0.8200   | 0.1600    | 0.1400 | 0.1500   | 2285              |
| **Random Forest**          | 0.8900   | 0.5000    | 0.0014 | 0.0027   | 2285              |
| **Neural Network (NN)**    | 0.8662   | 0.2040    | 0.0744 | 0.1091   | 2285              |
| **Convolutional Neural Network (CNN)** | 0.8900   | 0.0000    | 0.0000 | 0.0000   | 2217              |


## Confusion Matrix Summary

| Model                | True Negative (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) |
|----------------------|---------------------|----------------------|----------------------|---------------------|
| **Decision Tree**    | 16430              | 1639                | 1965                | 320                 |
| **Random Forest**    | 17930              | 3                   | 2214                | 3                   |
| **Neural Network**   | 17289              | 644                 | 2052                | 165                 |
| **Convolutional NN** | 17933              | 0                   | 2217                | 0                   |



## Findings

- Random Forest achieved an accuracy of **0.8901**, with a perfect precision of **1.0000**, but struggled to detect minority class instances, resulting in a recall of **0.0014** and an F1 score of **0.0027**. The confusion matrix shows it identified only **3 true positives** for the minority class while maintaining **17930 true negatives** for the majority class.

- Neural Network provided a test accuracy of **0.8662**, with a precision of **0.2040** and a recall of **0.0744** for the minority class. It correctly identified **165 true positives** for the minority class while achieving **17289 true negatives** for the majority class.

- Decision Tree achieved an accuracy of **0.82**, with a recall of **0.14** for the minority class. It predicted **320 true positives** while maintaining **16430 true negatives**, offering moderate performance in identifying the minority class.

- CNN delivered the highest test accuracy of **0.8877**, successfully classifying all **17933 majority class instances**. However, it failed to detect any minority class instances, resulting in a precision, recall, and F1 score of **0.00** for the minority class.



## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/diabetes-readmission
   cd diabetes-readmission
   
2. **Install Required Libraries**:
- pip install pandas numpy matplotlib seaborn tensorflow scikit-learn

3. **Run the Notebook**:
- jupyter notebook diabetes_readmission.ipynb

4. **Review Results**:
- Metrics and confusion matrices for each model.
- Visualizations for class distribution, feature importance, and more.
