# Lung Cancer Prediction

This project aims to predict the likelihood of lung cancer using various machine learning models and datasets. The main focus is on early detection to help increase survival rates by allowing patients quicker access to treatment.

- **Disclaimer:** This model is intended solely for educational purposes. If you wish to deploy it for real-world applications, it is highly recommended to fine-tune the model using a large and diverse dataset to ensure robustness, accuracy, and reliability in real-world scenarios.

This framing emphasizes the educational context while advising on the importance of further fine-tuning for practical use.

## Table of Contents
- [Overview](#overview)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Models Compared](#models-compared)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Future Work](#future-work)
## Overview

Lung cancer is a leading cause of cancer deaths globally. This project uses machine learning to predict the probability of lung cancer in patients based on medical data, including imaging and patient history.

## Datasets

We use several datasets, including both publicly available lung cancer patient data and simulated datasets. One of the prominent datasets used is the `lung_cancer_survey.csv`, which provides annotated lung nodule scans.

## Preprocessing

1. **Data Cleaning**: This step removes incomplete or noisy data.
2. **Feature Engineering**: Key features such as age, gender, smoking history are selected.
3. **Normalization**: StandardScaler is used to normalize the data to improve the performance of machine learning models.
4. **Splitting**: The dataset is split into training (73%), test (27%) sets.

## Models Compared

We evaluated several models to identify the best performing one for lung cancer prediction:

- **XGBoost (XGB):** A gradient-boosted decision tree algorithm known for high performance and speed. It is widely used in classification tasks.
- **Random Forest Classifier:** A robust ensemble method that builds multiple decision trees and aggregates their predictions.
- **Decision Tree Classifier:** A simpler, interpretable model that uses a tree-like structure to make decisions based on the features.
- **K-Nearest Neighbors (KNN):** A non-parametric method that predicts the class of a data point based on its proximity to other labeled data points in the feature space.

## Results

The following table summarizes the accuracy and F1-scores across different models:

| Model                     | Accuracy | F1-Score |
|---------------------------|----------|----------|
| Random Forest Classifier  | 95%      | 0.97     |
| Decision Tree Classifier  | 95%      | 0.97     |
| XGB Classifier            | 93%      | 0.96     |
| KNN Classifier            | 90%      | 0.95     |



## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/bhaskarachalla/Lung_Cancer_pred.git
    cd Lung_Cancer_pred
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
## Check Out the Live Web App

Just have a look at the website: [Lung Cancer Prediction Web App](https://lung-cancer-pred-scnb.onrender.com)

## Future Work

- **Hyperparameter Tuning**: Experiment with different architectures and hyperparameters to further improve accuracy.
- **Real-time Prediction**: Implement a real-time prediction feature that allows doctors to input patient data and get immediate risk analysis.




