# Machine Learning Breast Cancer Classification

This repository contains code for breast cancer classification using various machine learning models. The models are trained and evaluated on a breast cancer dataset to determine whether the person has cancer or not.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset. It consists of 569 instances with 30 features. The goal is to classify whether a breast tumor is benign (non-cancerous) or malignant (cancerous). The dataset is available at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer.

## Models

The following machine learning models are implemented and evaluated in this project:

1. LinearSVC
2. K-Nearest Neighbors (KNN)
3. Support Vector Machines (SVM)
4. Gradient Boosting
5. RandomForestClassifiers(RFC)


Each model is trained using the breast cancer dataset and tuned using hyperparameter optimization techniques.

## Evaluation Metrics

The best models are evaluated using the following metrics:

1. Accuracy: The ratio of correctly predicted instances to the total number of instances.
2. Precision: The ratio of true positive predictions to the sum of true positive and false positive predictions.
3. Recall: The ratio of true positive predictions to the sum of true positive and false negative predictions.
4. F1-Score: The harmonic mean of precision and recall.


5. The classification report, including precision, recall, F1-score, and support for each class, will be saved in the `classification_report.txt` file.

## Results

After evaluating all the models on the breast cancer dataset, the model that performed the best based on the metrics is the "RandomForestClassifier". It achieved the highest accuracy, precision, recall, F1-score, and using Confusion Matrix

## Conclusion

This project demonstrates the application of different machine learning models for breast cancer classification. By evaluating the models using various metrics, we determined the best performing model and provided a classification report for further analysis.



Happy classifying!
-vivek_raj