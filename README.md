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

## Data Source
UCI Machine Learning Repository - [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing) dataset.

## Final Report & Package Walk-Through

To reproduce this study, use modules in 'src' directory of this repo. (setup instructions below) and walk-through of the package is presented in the [final report](https://github.com/abhijitpai000/predictive_lead_scoring/blob/master/report/README.md)

## Setup instructions

#### Creating Python environment

This repository has been tested on Python 3.7.6.

1. Cloning the repository:

`git clone https://github.com/abhijitpai000/predictive_lead_scoring.git`

2. Navigate to the git clone repository.

`cd predictive_lead_scoring`

3. Download raw data from the data source link and place in "datasets" directory

4. Install [virtualenv](https://pypi.org/project/virtualenv/)

`pip install virtualenv`

`virtualenv lead_scoring`

5. Activate it by running:

`lead_scoring/Scripts/activate`

6. Install project requirements by using:

`pip install -r requirements.txt`

**Note**
* For make_dataset(), please place the raw data (bank-additional -> bank-additional-full.csv from data source) in the 'datasets' directory.
