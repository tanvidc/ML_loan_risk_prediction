## Evaluating Performance of Six Machine Learning Algorithms to Predict Creditworthiness of Bank Loan Customers

This was my very first ML project in 2016. 

Six Machine Learning Algorithms are used to predict the creditworthiness of loan seekers- Naïve Bayes, Linear Regression, Decision Trees, Random Forests, Extremely Randomized Trees and Support Vector Machines.  Parameters such as type of kernel, measure of information etc. are tuned to find the best performer. Predictions are cross validated by dividing data into training and test samples, accuracy of 70-75% is obtained. Relative feature importance gives an insight into characteristics of a person that would determine his/her borrowing and payment behavior. Algorithms are compared on the basis of computation time, stability, accuracy, F1 score and interpretability.

![numtrees](https://github.com/tanvidc/ML_loan_risk_prediction/blob/master/Report/numtrees.png)
Average error decreases as number of trees increase. After a threshold of about 25 trees there is no significant error reduction at the expense of computation.

![feature_importance](https://github.com/tanvidc/ML_loan_risk_prediction/blob/master/Report/feature_importance.PNG =100x)
Normalized coefficients of features shaded by importance. Numbers close to 1 in red represent direct relation while negative numbers- blue represent inverse relation between the feature and the predicted outcome- creditworthiness. White represents almost no correlation.

![acc_f1](https://github.com/tanvidc/ML_loan_risk_prediction/blob/master/Report/acc_f1.png)
Comparison of accuracy and F1 score for both 75-25 split of train and test set and 10-fold cross validation

![forest_d3](https://github.com/tanvidc/ML_loan_risk_prediction/blob/master/Report/forest_d3.png)
Decision tree truncated at depth = 3 as an example.
