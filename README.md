# CS_4641_Project
Final Project for CS 4641 Machine Learning Summer 2023
Machine Learning for Credit Card Fraud Detection

Credit card fraud detection is a valuable application of ML that has been researched extensively over the past decade [1]. The goal of researchers has been to use anonymized data supplied by credit card companies to train models that determine if a credit card transaction is fraudulent, generally as a binary classification [2]. A major goal is to allow for accurate real-time detection of credit card fraud that can prevent the unlawful purchases as they happen and notify the holder of the card of fraudulent activity [3]. Prior research has resulted in comparative analyses of various machine learning methods on the problem, as well as discussions about the main difficulties in designing an accurate classifier with the data supplied [2]. 

Despite the interest that has been taken in fraud detection, the problem of credit card fraud has gotten progressively more severe in the 2010s and early 2020s, with a projected $397 billion loss over the 2020s worldwide [1, 4]. This is in part due to the dynamic nature of the problem: as fraud detection improves, so too do fraudsters who often attempt to mimic the normal purchases of the card’s holder [2]. Around 65% of the total losses are covered by the credit companies, while the remaining 35% is left to the merchants to fill [4]. Improving the performance of models is a constant necessity.

We plan to use a combination of supervised and unsupervised learning for this. Supervised classifiers like Naive Bayes classification and SVMs can be used to provide confident classifications for an input as fraudulent or legitimate, while clustering methods are capable of clustering data into a cluster of legitimate purchases and one or more illegitimate purchases. Intuitively, we will start with hard clustering such as K-means and hierarchical techniques, as this is a binary classification, but we will also explore the use of soft boundaries: card holders can make unusual purchases, so it is important to determine whether a purchase is actually fraudulent or simply unusual. This gray area could be handled well by a Gaussian mixture model, where we could theoretically identify unusual purchases as somewhere between fraudulent and legitimate. We intend to use sk-learn implementations for these algorithms, although if we move to neural networks for better performance, we may make changes to the architecture for better results. Finally, one of the biggest issues with the problem is the imbalance of data: since there are far more legitimate purchases than fraudulent ones, there is a need for some approach to balance the training data for the model. We intend to make a comparison of various sampling methods to see what works best for each of our different models.

We hope to at least match the best performance on our imbalanced Kaggle dataset with as many methods as we can while making conclusions about the data with respect to its anonymized features. The main metric we will use, particularly given the imbalanced dataset, is balanced accuracy. This will especially help us minimize false positives.





Bib

1. https://www.inscribe.ai/fraud-detection/credit-fraud-detection

2. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8123782

3. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8776942

4. https://www.bankrate.com/finance/credit-cards/credit-card-fraud-statistics/#fraud

