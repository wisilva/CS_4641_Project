

CS 4641 Team 5 Project

Summer 2023

Machine Learning for Credit Card Fraud Detection

# Midterm Report

## Intro
Credit card fraud detection is a valuable application of ML that has been researched extensively over the past decade [1]. The goal of researchers has been to use anonymized data supplied by credit card companies to train models that determine if a credit card transaction is fraudulent, generally as a binary classification [2]. A major goal is to allow for accurate real-time detection of credit card fraud that can prevent the unlawful purchases as they happen and notify the holder of the card of fraudulent activity [3]. Prior research has resulted in comparative analyses of various machine learning methods on the problem, as well as discussions about the main difficulties in designing an accurate classifier with the data supplied [2]. 

Despite the interest that has been taken in fraud detection, the problem of credit card fraud has gotten progressively more severe in the 2010s and early 2020s, with a projected $397 billion loss over the 2020s worldwide [1, 4]. This is in part due to the dynamic nature of the problem: as fraud detection improves, so too do fraudsters who often attempt to mimic the normal purchases of the card’s holder [2]. Around 65% of the total losses are covered by the credit companies, while the remaining 35% is left to the merchants to fill [4]. Improving the performance of models is a constant necessity.

## Problem Definition
We used a combination of supervised and unsupervised learning for this. Supervised classifiers like Naive Bayes classification and SVMs can be used to provide confident classifications for an input as fraudulent or legitimate, while clustering methods are capable of clustering data into a cluster of legitimate purchases and one or more illegitimate purchases. Intuitively, we have started with hard clustering such as K-means and hierarchical techniques, as this is a binary classification, but we also explored the use of soft boundaries: card holders can make unusual purchases, so it is important to determine whether a purchase is actually fraudulent or simply unusual. We handled this case with a Gaussian Mixture Model, where we could identify unusual purchases as somewhere between fraudulent and legitimate. We intend to use sk-learn implementations for these algorithms, although if we move to neural networks in the final for better performance, we may make changes to the architecture for better results. 

Finally, one of the biggest issues with the problem is the imbalance of data: since there are far more legitimate purchases than fraudulent ones, there is a need for some approach to balance the training data for the model. To identify which transaction features were most important, we used principle component analysis and factor analysis. To address the problem of too few occurances of fraud, we downsampled the dataset. The dataset often listed features that were never collected, so we used k-nearest neighbors to deal with discrepencies. Below is a more thorough description of the data's problems and our solutions.

## Data cleaning
The data we used was taken from a competition for fraud detectionn hosted by Vesta, a company which works in that field. The data we used contains information about individual transactions which take place. Each transaction has an ID and label, after which it contains many features numerical and categorical features such as the amount, product code (for product in the transaction), card, address, time, and some features engineered by Vesta. The specific meaning of most of the values is often masked.

We cleaned this method in 2 ways.

Our data processing techniques attempt to resolve the 3 main issues with our raw data: an imbalanced dataset of primarily legitimate transactions, a large number of missing values, and mixed data (a combination of categorical and numerical features). We also were able to reduce the dimension of the data significantly.  First, we took 100,000 data points (approximately 1/6th of the total data set) from our data to use as the test set that does not see the models until test time, and we cleaned this set separately from the training data. 

The handling of missing data is done in data_processing.py, where we use a KNN imputation approach to fill in the missing values of a point with the mean of its 20 nearest neighbors. Due to the n^2 complexity of the KNN algorithm, performing the full algorithm on our dataset of size 490,500 x 392 is impossible. Instead, we perform an approximate KNN by performing the operation in batches of 5,000 points. This number allows for good performance while remaining possible to compute locally. It should be noted that the KNN imputer is only performed on the numerical features of the data; our approach to categorical data is handled later as we address feature extraction. Before the KNN imputation is performed, our numerical data is normalized to prevent differences in units from significantly affecting the distances. The test data is inferred separately.

Next, we performed feature extraction while simultaneously dealing with categorical features using Factor Analysis of Mixed data, an extension of PCA that uses a one hot encoder to address categorical data and allows for feature reduction of mixed data using the same method as PCA. After performing the FAMD, we selected the principal components that account for 95% of the variance and reduced used matrix multiplication to reduce our feature count from 392 to 159. At this point, all of our features are numerical, and the same principal components are used to reduce the dimension of the test data.

Finally, before passing the data train_data_clean_reduced_encoded.csv into our model, we downsample the majority class of unfraudulent transactions to some ratio of the fraudulent data. We commonly see best performance with a 1:1 ratio (20,000 samples of each class), which will be shown in some of the model evaluations later.

In conclusion, we begin with a sparse mixed dataset of size 590,500 x 392 and end with a full training set of size 490,500 x 159 and a full test set of size 100,000 x 159. Of the training set, we select between 40,000 and 490,500 data points based on performance, and we split that data into a validation set and a train set (or use cross validation).  

Additionally, we also cleaned the dataset using a similar method with a kNN model we wrote ourselves which capped the number of samples compared from the complete dataset rather than splitting the data into buckets before cleaning. This is expected to have little impact on features which were more complete & had lessing missing data, but may produce better results for filling in missing data for features that were missing a majority of the data. We ran this kNN method with k = 25 and a maximum complete sample size (sampled randomly from the complete samples) of 2500.

## Methodology

### Kmeans
The Kmeans algorithm clusters the data based on each point's distance to the cluster center. Because of this, the boundaries produced by kmeans are simple and based only on Euclidean distance. The data was cleaned with a k-nearest neighbors algorithm before being fitted to different
models and subjected to principal component analysis. Categorical features were one hot
encoded, and the dataset was downsampled to provide a balanced dataset. After this preparation, the slices of the dataset
were sent through a total of 15 kmeans clusterings. Each test used more features than the previous one, with the 15th test fitting the entire dataset. To prevent randomness and noise from polluting the results, each of the 15 tests were repeated 10 times, with the average results collected and graphed in the results section.
### GMM
GMM clusters the data by fitting a specified number of multivariate gaussian distributions to the dataset. Each point is then assigned a probability of being a member of one of the gaussian distributions. For the sake of visualization and comparison with other methods, each point was given a hard assignment to its most likely distribution.

The data was prepared identically to k-means, and a similar battery of tests were ran with
increasing numbers of principle components included. 15 tests using increasing numbers of features were ran, and each test
was ran ten times to minimize the effects of random starting conditions. For the sake of rigor and
completeness, a gaussian mixture of 2 components and 4 components were both tested. In
a round of preliminary testing, the average log likelihoods of gaussian mixture models with
various numbers of components were collected, and 4 clusters was identified as the best
performer via the elbow method.\
### DBSCAN
The DBSCAN algorithm creates clusters by separating regions of lower from regions of higher density. A cluster found with DBSCAN is defined as a maximal set of density-connected points. The cluster will represent a group of data points that are believed to be statistically similar. Additionally, the results of DBSCAN have two hyperparameters, epsilon, the allowable distance between points in the cluster, and minimum point, the minimum amount of points required to be its own cluster. However, almost tuning is done by adjusting epsilon. 

The data was prepared identically similar to Kmeans and GMM. However, instead of using pca, we used TNSE, which has shown to work more effectively then pca. TNSE is tuned with one hyperparameter, perplexity. TNSE was used to reduce the data to two dimensions, which was needed DBSCAN to work effectively as it heavily relies on Euclidean distance. We ran TNSE for 20 thousand iterations for all perplexities 5 and 50, and then chose the dataset produced by TNSE that appeared to exhibit clustering the most from looking at the graphs produced. Then, we ran TNSE on that dataset with the best and second-best perplexity for 100 thousand iterations. The dataset produced was then run through DBSCAN with an various epsilon and minimum cluster sizes and were tuned from there. The results were then visualized showing the different clusters. They then analyzed to determine if any of the meaningfully sized clusters differed from the average 3.5% to see if they provided any value for real-world applications.

### Naive Bayes
The Naive Bayes Classifier is our first supervised technique and only supervised classifier for the midterm checkpoint. Naive Bayes fits probability distrubutions, in this case Gaussian Distributions to the dataset. The model assumes each feature is independent, and it examines the likelihood of each feature of a given transaction to give a likelohood that the transaction is fraud. 

With the Naive Bayes (Gaussian) classifier, it is especially interesting to look at how an imbalance in the training set will affect the algorithm, as the priors are calculated immediately from the data itself. I will run the algorithm at multiple ratios of MajoritySet:MinoritySet (legitimate transactions and fraudulent transactions respectively) to see how this affects the various evaluation scores of the model.

## Results
After cleaning the data, the three highest variance features were identified via Principle Component Analysis. The graph below graphs the data points as a function of those components. The yellow points are fraudulent cases.

<img src="images/GT.png" width="500">

### Kmeans

Over the numerous tests, the k-means algorithm
had an average balanced accuracy of 0.59 and an average F-Measure of 0.32. These results
show that the simple boundaries of k-means did not give a good representation of the highly
nonlinear data. The model's convergence to high error results implies that k-means is not capable of
adequately modeling the data.\
Below is the average Rand Statistic of the kmeans clustering with varying fractions of the features concerned. The horizontal axis shows what portion of the features were considered, starting with one fifteenth of the data and ending with the whole dataset. The Rand Statistic is similar to normal accuracy, so it is not very reliable for unbalanced datasets. However, the downsampling solves this problem. The result of about 0.6 shows that the clusters were mostly identifying true negative cases, which is not very useful in commercial applications. The repetition of each test ten times was effective in getting rid of noise due to random initialization, but the inclusion of more features was unable to make the model improve its Rand Statistic.

![kRS](images/kRS.png)

These graphs show the performance of kmeans as measured with precision, recall, f-measure, and balanced accuracy. The results for precision seem impressive, but f-measure is a more informative metric because it accounts for the model's precision and Recall. Similarly, balanced accuracy is more informative than regular accuracy. These results show that the model's performance was mostly invariant as more features were added. The lack of convergence is troubling because the model is getting stable results, but poor ones. For instance, the balanced accuracy never went above 60.11%.

![gmm2Clusters](images/kOther.png)
The Jaccard Coefficient ignores the contribution of true negative results, so the dismally low values show that a large portion of the accuracy found above was due to the model correctly identifying cases of legitimate transactions. In the case of real world applications, there is less value in identifying true legitimate cases than true fraudulent cases because it may only take one fraudulent user to steal large sums of money. With this real-world factor in consideration, the average Jaccard Coefficient of 0.19 proves that kmeans is functionally useless for real-world applications.
![gmm2Clusters](images/kJC.png)
The graph below plots the ground truth clustering on the left, and the kmeans clustering on the right. The unimpressive overlap of the two graphs illustrates the point made by the pairwise measures. 

<img src="images/K (2).png" width="1000">

### GMM
The binary gaussian mixture had an average balanced
accuracy of 0.67 and an average F-Measure of 0.56. The 4-component mixture had an
average balanced accuracy of 0.56, and an average F-Measure of 0.24. Overall, the
gaussian models did not show a sufficient improvement over k-means. Their convergences
to high error results imply they the gaussian mixture model is not capable of adequately
modeling the data.\
Below is the average Rand Statistic of the kmeans clustering with varying fractions of the features concerned. The horizontal axis shows what portion of the features were considered, starting with one fifteenth of the data and ending with all the data. The rand statistic shows that, even with the help of lots of false negatives to boost its score, the accuracy of the model is quite poor. GMM gives soft assignments, so the pairwise metrics were evaluated based on the most likely assignment for the sake of consistent comparison with kmeans clustering.

![gmm2Clusters](images/gmm2RS.png)

The high precision measure shows that each cluster had a high degree of purity. The stability of this value shows that the model had a robust resilience to preventing too many false positives from contaminating the fraud cluster. However, the low recall values show that the false negative rate was quite high. F-measure is the harmonic mean of precision and recall, so the low recall values dragged that score down to unacceptable levels. Balanced accuracy showed a similar trend of stable, poor results. Because the downsampling during data cleaning solved the problem of an imbalanced dataset, the balanced accuracy was not meaningfully different from the rand statistic.

![gmm2Clusters](images/gmm2PrecisionRecall.png)

![gmm2Clusters](images/gmm2FMeasure.png)

The jaccard coefficient further reveals the problem of false positives. The accuracy and f-measure results are propped up by the large amount of true negative findings, but when looking just at the accuracy of true positive results, the key weakness of this clustering is found. The jaccard coefficient for GMM was better than that of kmeans, but it still had a dismal average result of 0.44, which is still far below what would be needed for any real-world application.

![gmm2Clusters](images/gmm2JC.png)

The graph below plots the ground truth clusters on the left and the predicted clusters on the right. Even though the graph only shows the top three principal components, the poor correspondence of this graph to the ground truth one shows that this clustering lacks much predictive power.

<img src="images/G2 (2).png" width="1000">

Although the task of identifying fraud is a binary clustering problem, the poor results above show that the previous models were too simple to represent the data. Before addressing this problem with more complex models, it is worth examining a more complex Gaussian Mixture Model with four clusterings instead of two. In this case, the cluster with the highest proportion of fraud cases is taken to be the fraud cluster, and the other clusters are treated as legitimate.\
However, adding more clusters did not help the performance in rand statistic. The smaller clustering had virtually no effect, and the results are almost identical to the binary clustering case.

![gmm2Clusters](images/gmm4RS.png)

The additional clusters only exacerbated the existing behaviors of the binary case. Dividing the data set up into smaller parts left the true fraud cluster smaller in the 4-mixture case. This smaller cluster let the 4-mixture GMM reach exceptionally high precision results because the contamination of the cluster was so small. However, the smaller cluster size lead to far more false negative cases, which drive the average recall to below 0.2. The increased precision could not outweigh the dismal recall, so the f-measure was the worst we have encountered so far.

The smaller clustering did not have much meaningful impact on the balanced accuracy, because the additional false negatives were counterbalanced by a decrease in false positives. The balanced accuracy was mostly unchanged from the binary mixture model, which was also not very good. 

![gmm2Clusters](images/gmm4Other.png)

The true weakness in the 4-cluster model is shown by the Jaccard Coefficient. Adding two additional clusters made the resulting clusters smaller than their binary counterparts, which increased the number of false negatives in the data. These false negatives propped up the balanced accuracy shown above. By neglecting these additional false negatives, this model shows the worst performance so far, with an average jaccard coefficient of 0.13. The previous models were too simple to represent the data well, but the added complexity will not come from simply adding more clusters.

![gmm2Clusters](images/gmm4JC.png)

Here is the clustering of the 4-gaussian mixture. The colors represent the highest likelihood assignment of each point. The first image shows all resulting clusters, and the second image isolates the fraud cluster.

<img src="images/G41 (2).png" width="1000">
<img src="images/G42 (2).png" width="1000">

Overall, the gaussian mixture model was unsuccessful at producing anything that would be useful in a real-world application. We make one final attempt at an unsupervised fraud detection method before moving on to the more powerful supervised methods. 



### DBSCAN
Our group's implementation of DBCAN involved using the same preprocessing pipeline used with the rest of the implementations. However, we did not use PCA to reduce the dimensionality of the data to two axes because we knew that DBCAN only works well with two dimensions or less. We used TSNE instead, which has shown to be an improvement over PCA in numerous cases and we thought it fit our case. Due to the computationally intensive nature of DBSCAN, we decided to use the GPU-accelerated version of SKLEARN, CUML, on colab. This allowed us to run the many perplexities and iterations of TSNE that are needed to evaluate, which perplexity we should use for DBSCAN as well as whether our number of iterations was enough to reach stability for TSNE. We ran all perplexity values, the hyperparameter for TSNE, from 5 to 50, which are the recommended values, for 10 thousand iterations. Then we ran that set of perplexities again for 20 thousand iterations, which is far above the default value of 1 thousand for TNSE. We did this because when we were running with the default number of iterations we were not reaching stability as a lot of the perplexities had pinching. Even with 20 thousand iterations, still weren't getting much obvious clustering, so we decided to choose the best-looking group with no pinching and looked like it was separating into 2 groups, perplexity 46, since we would run out of compute time if we tried running all perplexities for more iterations. We ran a perplexity of 46 for 100 thousand iterations and then ran GPU-accelerated DBSCAN on the dimensionally reduced dataset with eps ranging from 1000 to 4000 with a min sampling size of 4, 10, or 15. This provided DBSCAN models that either had far too many clusters in the model to be useful or just had a single large cluster and few smaller ones that were not statistically significant. We still did an analysis of the DBSCAN models that looked most promising. The clusters that looked most promising were had hyperparameters tuple of eps and samples size that were (2000, 15) and (1500, 15). We examined whether the percentage of fraud in any of the clusters was a lot higher or lower than the mean of roughly 3.5 percent. After examining all the clusters, it was found that all the clusters that looked promising were part of a cluster that had around 99 percent of the data points and the mean fraud was still around 3.5 percent. I then decided to use the data provided by running TNSE with a perplexity of 50 with 100 thousand iterations as it provided a different-looking group than the perplexity of 46 data. The model still encountered the same problem and we still analyzed the most promising DBSCAN models with the hyperparameters tuple of eps and samples size that were (5000, 2), (2000, 4), (2000, 3), (2000, 2), (1000, 4), (750, 4). However, they still provided the same non-applicable models.

#### TNSE Analysis
##### All perplexities from 5 to 50 with 20 thousand iterations
![TNSE](images/TNSE.png)
From the visualization of the data after the dimension reduction via TNSE, the data that was dimensionally reduced with a perplexity of 46 was chosen because it seemed to exhibit the most distinct clustering.

##### Perplexity 46 with 100000 iterations
![TNSE_50_Perp_100000_iters](images/TNSE_50_Perp_100000_iters.png)
This is the dataset produced after running TNSE with a perplexity of 46 for a hundred thousand iterations. We decided to run it for the additional iterations because it did not look like the dataset was done clustering at 20 thousand iterations. 

##### Perplexity 50 with 100000 iterations
![TNSE_50_Perp_100000_iters](images/Perp_50_100000_Iters.png)
This is the dataset produced after running TNSE with a perplexity of 50 for a hundred thousand iterations. We decided to run it for the additional iterations because it did not look like the dataset was done clustering after 20 thousand iterations. 

#### DBSCAN Analysis
##### DBSCAN of data of perplexity 46 with 100 thousand iterations
![DBSCAN_Perp_46](images/perp_46_DBSCAN.png)
The DBSCAN results from a TNSE perplexity of 46 were unsuccessful at producing a model that would have a real-world application. The data either produced models with far more clusters than would be useful with a usually single large cluster and many small clusters that were statistically insignificant. We still did an analysis of the DBSCAN models that looked most promising. We examined whether the percentage of fraud in any of the clusters was a lot higher or lower than the mean of roughly 3.5 percent. After examining all the clusters, it was found that all the clusters that looked promising were part of a cluster that had around 99 percent of the data points and the mean fraud was still around 3.5 percent. We examined the clusters with hyperparameters tuple of eps and samples size that were (2000, 15) and (1500, 15) and they all exhibited this behavior. You can also see that the number of clusters increases with a larger amount of minimum samples and a smaller eps as expected.

##### DBSCAN of data of perplexity 50 with 100 thousand iterations
![DBSCAN_perp_50](images/perp_50_DBSCAN.png)
After the unsatisfactory results from the dataset produced by a TNSE perplexity of 46, we decided to use the dataset produced from a perplexity 50 which was markedly different from the perplexity of 46. The results were almost identical to the results from a perplexity of 46. We examined the clusters with hyperparameters tuple of eps and samples size that were (5000, 2), (2000, 4), (2000, 3), (2000, 2), (1000, 4), (750, 4) and they all exhibited the same nonreal world applicable behavior.

### Naive Bayes

Below are the accuracy, balanced accuracy, and F1 score of the model graphed as a function of fraud: legitimate transactions.

![accuracy](images/acc_ratio.png)

![balanced_accuracy](images/balanced_ratio.png)

![f1_score](images/f1_ratio.png)

The plots here indicate that the model rapidly begins to overfit as the dataset becomes unbalanced. The F1 score, which measures the harmonic mean of precision and recall, indicates that the model becomes increasingly skewed towards labeling all data as legitimate, which could indicate especially that the Bayesian Priors are having a significant effect on the fitting of the model.

This is somewhat surprising to us, as we suspected that there would be some benefit to the skewed Priors for unbalanced data. This relationship was not supported by the performance of the model.


As F1 score is particularly useful for measuring the performance of a classifier with imbalanced data, this should be considered far more heavily than the accuracy, which likely indicates that the model is rapidly overfitting.

Fixing this ratio at 1, we test the performance of our model on the test set, which was cleaned separately.

![test_performance](images/test_performance.png)

Even with a balance between the two class labels in the training set, the F1 score of the data on the test set is low. It is possible that simply more data is required to improve the performance of the Naive Bayesian Classifier. As such, we might explore upsampling techniques in the future to balance the data while providing more samples. This, like many things, is difficult in high dimensions, our current FAMD approach to data representation should help with this matter.  





Link to our semester plan: https://docs.google.com/spreadsheets/d/1Jp_Bu6QtXSaUK9Z2fSecP_BQxmAvKgUN0jklrbbOljo/edit?usp=sharing

Link to our dataset: https://www.kaggle.com/competitions/ieee-fraud-detection/data

Responsibilities:

Sam: Assist with data cleaning, sampling, and feature reduction. Implementation, testing, and visualization of DBSCAN, Midterm Report

William: Assist with data cleaning, sampling, and feature reduction. Implementation, testing, and visualization of Hierarchical Clustering and SVM, Proposal and Github management

Stefan: Assist with data cleaning, sampling, and feature reduction. Implementation, testing, and visualization of Neural Networks and Naive Bayes

Carter: Primary for data cleaning, sampling, and feature reduction. Implementation, testing, and visualization of Regressions

Keyes: Assist with data cleaning, sampling, and feature reduction. Implementation, testing, and visualization of K-Means and GMM, Midterm Report

All: Results comparison and final report, peer reviews.


Bibliography

1. “Credit Card Fraud Detection: Everything You Need To Know.” Credit Card Fraud Detection: Everything You Need to Know, 13 May 2023, www.inscribe.ai/fraud-detection/credit-fraud-detection.

2. Awoyemi, John O., et al. Credit Card Fraud Detection Using Machine Learning... - IEEE Xplore, ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8123782. Accessed 17 June 2023. 

3. Thennakoon, Anuruddha, et al. Real-Time Credit Card Fraud Detection Using Machine Learning - IEEE..., 2019, ieeexplore.ieee.org/document/8776942. 

4. Egan, John. “Credit Card Fraud Statistics.” Bankrate, 12 Jan. 2023, www.bankrate.com/finance/credit-cards/credit-card-fraud-statistics/#fraud.

