# Basic Information
  This repository is used for storing Python codes of my machine learning assignment.
## asg3
The code in **asg3** package is an implementation of Neural Network's **back propagation algorithm**. The process including preprocessing data, training data and making predication of test data. Also, I have implemented **sigmoid**, **tanh** and **relu** activation functions and used three different dataset to compare the performance among these three activation functions.
**Note**: the codes were based on sample code given by Professor Anurag Nagar, especially the starting code for **back propagation** part. Thanks for his work.

## asg4
The code in **asg4** package is an implementation of **Multinomial Naive Bayes** classifier to real world text dataset.
I used the **amazon_cells_labelled** dataset to run the program. According to the result, the precision, recall and f1-score evaluation metrics are both 0.81. This result was achieved with an additive smoothing parameter of 0.8 and the fit_prior parameter of False. The precision is 0.81, which means almost 80 percent of the predictions are correct. The recall is 0.81, which means almost 80 percent of all positive(here is label ‘1’) sentence are predicted correctly. 
I think this result can be improved by trying more values of smoothing parameter, and more k values of k-fold cross validation.
## asgkmeans
The code in **asgkmeans** package is an implementation of **Kmeans Algorithm** to cluster tweets by utilizing **Jaccard Distance** metric.
## project
The code in **project** package is to solve the **Kaggle** competition: **[TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)**.





