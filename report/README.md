# Final Report
# Predictive Lead Scoring using ML

### Overview:
Predictive Lead Scoring is a method used to analyze lead behavior in historical customer data to find patterns resulting in a positive business outcome, such as a closed deal with a client. In this study, I developed a lead scoring model using the [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing) dataset, which contains the outcome of clients subscribing to a term deposit or not based on a direct marketing campaign performed by a Portuguese bank.

**Model Design:**

Since the evaluation of a classification model is tricky, I assumed that Losing a potential customer cost > Sales Resource Cost as a business objective, which statistically translates to developing a model that gives Low False Negatives and High True Positives, with balancing False Positives.

**Model Outcome:**

To mimic a real-time model evaluation, I separated ~10,000 observation points from the dataset and trained on ~30,000 observation points. The following is the result of my trained LightGBM model on the hold-out dataset.

>  75.04% of Leads predicted by the model have resulted in conversion. And, a 29.56% False Positive rate, 24.95% False Negative Rate is observed.


*Segmenting Leads based on Model Predictions:*


 <img src="https://github.com/abhijitpai000/predictive_lead_scoring/blob/master/report/figures/output_20_1.png" width="500" />

### Findings:
Based on Feature Importance and Permutation Importance, the following features have a significant impact on training and target predictions.
* nr.employed = number of employees 
* age = age of the client
* campaign = number of contacts performed during this campaign and for this client
* euribor3m = euribor 3 month rate 

<img src="https://github.com/abhijitpai000/predictive_lead_scoring/blob/master/report/figures/feature_importance.png" width="600" />

### About Bank Marketing Dataset.
This dataset contains outcome of clients subscribing to a term deposit or not based on a direct marketing campaign performed by a Portuguese bank.

**Source** UCI Machine Learning Repository [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing) dataset.
 


# Analysis Walk-through

**Table of Contents**
1. [Experiment](#experiment)
1. [Package Introduction](#introduction)
2. [Preprocess](#preprocess)
3. [Training](#train)
4. [Test Prediction](#predict)
5. [Lead Scoring](#scoring)

# Experiment: <a name="experiment"></a>

* *Model Selection :* 

I experimented with four machine learning algorithms, Logistic Regression, Random Forest Classifier, XG Boost Classifier, and LightGBM Classifier. Based on stratified k-fold cross-validation, evaluated using ROC_AUC, Average Precision & Precision-Recall, LightGBM predictions were in line with the business objective.



* *Fixing Class Imbalance :* 

The dataset has a class imbalance of 89:11 (Negative(0):Positive(1)). To fix the imbalance I experimented with Random Under Sampling, Random Over Sampling & Class Weights Balanced techniques. Out of the 3, I selected the Class Weight Balanced method, as it provides no information loss, has fairly optimum training time & was compliant with the model design choice.


```python
# Changing Current Working Directory.

mydir = "Git\Clone\Path"

%cd $mydir
```

# Package Introduction <a name="introduction"></a>

**Codebase Structure** 'src' directory.

| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| preprocess | make_dataset() | Performs pre-processing | raw_file_name | train_set, train_clean, test_set, test_clean & ord_encoder.pkl | train_clean, test_clean
| train | train_model() | Trains LightGBM Model on train_clean | -- | lead_scoring_model.pkl | training cross_validation results.
| predict | test_model() | Predicts on the test_clean & Segments Leads into High, Medium, Low Probability Categories based on model prediction  | thershold=0.35 | lead_scoring.csv | classification_report, confusion_matrix(normalize="true")


```python
# Local Imports.
from src.preprocess import make_dataset
from src.train import train_model
from src.predict import test_model

# Analysis.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Joblib
import joblib
```


```python
# Loading Raw Dataset.

raw = pd.read_csv("datasets/raw.csv")

raw.shape

"""
    (41188, 21)
"""
```









```python
# Checking Top 4 Rows, TARGET = "y", (No = 0, Yes = 1)

with pd.option_context("display.max_rows", 4, "display.max_columns", 25):
    display(raw.head(4))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>261</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>149</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>226</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>admin.</td>
      <td>married</td>
      <td>basic.6y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>151</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Target Class Weights Balance.

raw.y.value_counts(normalize=True)

"""
    no     0.887346
    yes    0.112654
    Name: y, dtype: float64
"""
```








# Data Pre-processing <a name="preprocess"></a>

**make_dataset()** Based on the insights gained from experiment, make dataset will perform following actions.
* Drops "duration" feature, which is the last contact duration, in seconds. Since this dataset is based on a Direct marketing campaign done through phone calls, the duration is not known prior to the campaign and the outcome will be known when the call ends, this feature could pose a potential target leakage, may result in over-optimistic predictions.
* Splits data in train_set, test_set.
* Ordinal Encodes Categorical features of both train and test_set yielding train_clean and test_clean.


```python
# Pre-processing Data.

train_clean, test_clean = make_dataset(raw_file_name="raw.csv")

train_clean.shape, test_clean.shape

"""
    ((30891, 20), (10297, 20))
"""
```









```python
# Top 4 rows of Train_Clean.

with pd.option_context("display.max_rows", 4, "display.max_columns", 25):
    display(train_clean.head(4))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>poutcome</th>
      <th>age</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>52</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>1.4</td>
      <td>94.465</td>
      <td>-41.8</td>
      <td>4.961</td>
      <td>5228.1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>40</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>-1.8</td>
      <td>93.075</td>
      <td>-47.1</td>
      <td>1.405</td>
      <td>5099.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>52</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.858</td>
      <td>5191.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


# Train Model <a name="train"></a>

**train_model()**
* Fits a LightGBM model to train_clean dataset.
* Performs 10 folds Cross-validation.
* Returns Training Cross-validation results.



```python
# Train Model.

cv_results = train_model()
```


```python
pd.DataFrame(cv_results)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fit_time</th>
      <th>score_time</th>
      <th>test_roc_auc</th>
      <th>test_precision</th>
      <th>test_recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.256551</td>
      <td>0.028591</td>
      <td>0.827517</td>
      <td>0.385502</td>
      <td>0.670487</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.244275</td>
      <td>0.030011</td>
      <td>0.798103</td>
      <td>0.374795</td>
      <td>0.656160</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.289112</td>
      <td>0.033473</td>
      <td>0.797442</td>
      <td>0.354785</td>
      <td>0.617816</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.239812</td>
      <td>0.029729</td>
      <td>0.820130</td>
      <td>0.388795</td>
      <td>0.658046</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.287818</td>
      <td>0.030424</td>
      <td>0.803629</td>
      <td>0.391534</td>
      <td>0.637931</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.242162</td>
      <td>0.030226</td>
      <td>0.784969</td>
      <td>0.364865</td>
      <td>0.620690</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.269596</td>
      <td>0.030200</td>
      <td>0.802204</td>
      <td>0.369863</td>
      <td>0.620690</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.250025</td>
      <td>0.026363</td>
      <td>0.784706</td>
      <td>0.345277</td>
      <td>0.609195</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.264831</td>
      <td>0.027636</td>
      <td>0.804357</td>
      <td>0.369492</td>
      <td>0.626437</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.246539</td>
      <td>0.022438</td>
      <td>0.780845</td>
      <td>0.372881</td>
      <td>0.632184</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Training Scores.

print(f"\nTRAIN SCORES:"
          f"\nROC_AUC: {np.mean(cv_results['test_roc_auc'])}"
          f"\nPrecision: {np.mean(cv_results['test_roc_auc'])}"
          f"\nRecall: {np.mean(cv_results['test_recall'])}")

"""
    TRAIN SCORES:
    ROC_AUC: 0.8003901184771813
    Precision: 0.8003901184771813
    Recall: 0.634963607021704

"""
```

    

    

# Test Prediction <a name="predict"></a>

**test_model()**
* Predicts on the test_clean dataset.
* Categorizes Leads into high, medium, low probability based on the model predictions
* Returns Classification Report and Confusion Matrix(normalized="true")



```python
# Testing Model.

report, conf_mx = test_model()
```


```python
# Test Scores.

print(f"\nTEST RESULTS:"
          f"\n {report}"
          f"\nTrue Positive Rate: {round(conf_mx[1][1]*100, 3)}"
          f"\nFalse Positive Rate: {round(conf_mx[0][1]*100, 3)}"
          f"\nFalse Negative Rate: {round(conf_mx[1][0]*100, 3)}")
          
"""
    TEST RESULTS:
                   precision    recall  f1-score   support
    
               0       0.96      0.70      0.81      9139
               1       0.24      0.75      0.37      1158
    
        accuracy                           0.71     10297
       macro avg       0.60      0.73      0.59     10297
    weighted avg       0.88      0.71      0.76     10297
    
    True Positive Rate: 75.043
    False Positive Rate: 29.566
    False Negative Rate: 24.957
"""
```

    

    

# Lead Scoring <a name="scoring"></a>

* Based on the Decision Thershold of 0.35 selected using ROC-AUC Curve & PR Curve, model predicts a decision of "high", "medium", "low" probability for each customer in the test_set.


```python
lead_scoring = pd.read_csv("datasets/lead_scoring.csv")
```


```python
# Top 4 rows of Train_Clean -> Class_1_prob, Predicted Probability.

with pd.option_context("display.max_rows", 4, "display.max_columns", 25):
    display(lead_scoring.head(4))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>poutcome</th>
      <th>age</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>ground_truth</th>
      <th>class_1_prob</th>
      <th>predicted_probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>39</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>1.4</td>
      <td>93.918</td>
      <td>-42.7</td>
      <td>4.957</td>
      <td>5228.1</td>
      <td>1</td>
      <td>0.333432</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>55</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>-1.8</td>
      <td>93.075</td>
      <td>-47.1</td>
      <td>1.405</td>
      <td>5099.1</td>
      <td>0</td>
      <td>0.450695</td>
      <td>high</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>39</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>1.4</td>
      <td>94.465</td>
      <td>-41.8</td>
      <td>4.961</td>
      <td>5228.1</td>
      <td>0</td>
      <td>0.249140</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>56</td>
      <td>7</td>
      <td>999</td>
      <td>0</td>
      <td>1.4</td>
      <td>93.444</td>
      <td>-36.1</td>
      <td>4.963</td>
      <td>5228.1</td>
      <td>0</td>
      <td>0.325643</td>
      <td>medium</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Categorizing Probabilities predicted by model.

plt.figure(figsize=(10, 8))
sns.countplot(lead_scoring["predicted_probability"])
```




    <AxesSubplot:xlabel='predicted_probability', ylabel='count'>




![lead_cats](https://github.com/abhijitpai000/predictive_lead_scoring/blob/master/report/figures/output_20_1.png)


**END**
