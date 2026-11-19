
# Train/Test Split and Model Evaluation

## Project Overview

This project implements a Logistic Regression classification model using the Iris dataset and evaluates its performance using a train/test split.

## Objectives

- Load and explore a dataset.
- Divide data into training and testing sets.
- Train a Logistic Regression model.
- Predict outcomes using test data.
- Evaluate model performance using classification metrics.

## Dataset

Iris Flower Dataset from Scikit-learn.

- 150 samples
- 4 numerical features
- 3 target classes: Setosa, Versicolor and Virginica.

## Train/Test Split

The dataset is divided into:
- 80% Training Data
- 20% Testing Data

Stratified splitting is used to maintain class proportions.

## Algorithm

Logistic Regression is a supervised machine learning classification algorithm used to predict categorical outcomes.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run

Install dependencies:

pip install -r requirements.txt

Run:

python main.py

## Results

Evaluation metrics and visualization graphs are automatically generated inside the outputs folder.

## Conclusion

The Logistic Regression model is trained using the training dataset and evaluated on unseen testing data. Its performance is measured using multiple classification metrics.
