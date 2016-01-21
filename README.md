# regression-w-unknown-feature-names

We have in hand a regression problem with 5000 observations and 254 features, whose names are not known. The number of features is fairly large and we are expected to report on the important features. Therefore we it will be sensible to use a model that appropriatly regularize the outcome and automatically perform 'feature-selection'. ***Lasso*** is perfect for that as it uses an ***L1 norm penalty*** on the coefficient vector and promotes sparsity.

In addition to Lasso, we performed ***random forest regression***, which outperformed Lasso on the training set. To compute the optimal regularization parameter for Lasso, we empleyed a ***cross validation*** approach. Similarly, we determined optimal number of trees and number of features for the random forrest regresssor via a small scale ***grid search*** powered by cross validation.

Random forrest classifer also gives importance score for each feature in terms of average reduction in the variance for each cut in that feature. We also report on the ***importantance of features as determined by random forrest regressor***, and compare those with the ones selected by Lasso. 

Other feature selection methods such as forward selection where features are added one at a time could have been used. We will omit more complex feature engineering methods such as considering higher order terms and correlation analysis to add new features to incorporate interactions.

Some comments on the qualty of the data set and the data cleaning methods that were employed are given next.

The notebook can be accessed [here](http://nbviewer.jupyter.org/github/osemer01/regression-w-unknown-feature-names/blob/master/regression_wo_knowing_feature_names.ipynb): 
