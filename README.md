Telco Customer Churn Prediction
Objective

Predict whether a customer will churn (Yes/No) using the Telco Customer Churn dataset.

The model outputs:

1 → Yes (Customer will churn)

0 → No (Customer will not churn)

Train / Validation Split

We used an 80/20 stratified train-validation split to preserve the original churn distribution (~26% churn rate).

Stratification ensures both churn and non-churn customers are proportionally represented in training and validation sets.

Data Cleaning

Converted TotalCharges to numeric (handled blank values)

Removed rows with missing values

Dropped irrelevant column (customerID)

One-hot encoded categorical variables

Converted target variable (Churn) into binary format (Yes=1, No=0)

Exploratory Data Analysis (EDA)

Key observations:

The dataset is moderately imbalanced (~26% churn).

Customers with month-to-month contracts churn more frequently.

Higher MonthlyCharges are associated with increased churn probability.

Customers with longer tenure are less likely to churn.

Contract type and tenure are strong predictors of churn behavior.

Model Selection
Final Model – Random Forest (Recall-Optimized)

Hyperparameters:

n_estimators = 200

max_depth = 10

class_weight = balanced

random_state = 42

Class imbalance was handled using class_weight="balanced" to improve churn detection.

Metrics Reported

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

Classification Report

Validation Results

Accuracy: ~0.83

Churn Recall: 0.94

F1 Score (Churn): ~0.74

False Negatives: 109

Confusion Matrix
[[4046 1117]
 [ 109 1760]]


Interpretation:

1760 churn customers correctly identified.

Only 109 churn customers were missed.

Higher false positives (1117), but significantly improved churn detection.

Error Analysis

The model prioritizes recall for churn customers.

Compared to an accuracy-focused model, this approach:

Reduces missed churn customers from 725 to 109.

Improves churn recall from 0.61 to 0.94.

Slightly reduces overall accuracy.

This trade-off is beneficial in real-world scenarios where missing a churn customer is more costly than incorrectly targeting a non-churn customer.

Final Conclusion

The recall-optimized Random Forest model was selected as the final model because it significantly improves churn detection while maintaining strong overall performance.

From a business perspective, accurately identifying customers at risk of churning enables proactive retention strategies and reduces revenue loss.