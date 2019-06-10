# CategoricalNaiveBayes

This implementation of the Naive Bayes Algorithm eliminated the need for one hot encoding of categorical features in a dataset. It does not incorporate Gaussian Naive Bayes however and only handles categorical data. A threshold hyperparameter has also been added allowing the user to tweak at what percent classification occurs at.

This implementation is still a bit unstable. One known issue is that if the training set is missing any of the categories, it will fail.

*This project was completed as part of an Independent Study at Northeastern Illinois University under Professor Xiwei Wang. It was created mostly for my own learning purposes, but performed as well as the SciKit-Learn Implementation of Bernoulli Naive Bayes, which is usually used for categorical data.*
