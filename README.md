# Project Overview
The aim of this project is to evaluate and compare the sentiment prediction of various supervised custom machine learning (ML) algorithms and an unsupervised pre-trained sentiment prediction model. In this project, we use Kaggle's [Amazon fine food reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset for the analysis. The supervised custom ML algorithms used in this project are traditional ML algorithms namely Logistic Regression, Multinomial Naive Bayes, SVM and LightGBM, and deep learning algorithms namely deep Neural Network and 1 dimensional CNN. For the pre-trained sentiment prediction model, a version of Bidirectional Encoder Representations from Transformers (BERT), which is fine-tuned for sentiment analysis of product reviews, is used. Here is the link to the notebook of the project.
# Dataset
The dataset consists of 568,454 user reviews on various food products. For each record, the dataset consists of three important data inputs (columns): review, summary and rating. The review describes the user's feedback regarding the product while the summary describes the general sentiment of the feedback. The rating of each review is given in numbers between 1 and 5, 1 and 5 being the poorest and the best respectively.
# Methodology
We aim to implement a model that will predict a review's sentiment as positive, neutral or negative. To achieve the objective, the following key tasks were performed;
   1. Data cleaning to remove redundant reviews and errors associated with other review attributes.
   2. Categorizing review ratings 1-2, 3 and 4-5 as negative, neutral and positive sentiments respectively.
   4. EDA was performed to analyse the distribution of data per rating/sentiment as well as the word cloud of the dataset per sentiment category.
   5. Selecting a small subset of unseen data from the dataset for evaluating and comparing the performances of the models.
   6. Balancing the remaining dataset by selecting a subset of reviews consisting of an equal number of reviews for each sentiment category.
   7. Merging review and summary columns to form a new data input column.
   8. Text pre-processing of the merged data.
   9. Splitting the dataset into train and test datasets.
   10. Modelling the supervised models with the train data and evaluating them with the test dataset to find the best-performing model. TFIDF and Glove6B's vector representation methods were used for text vectorization for traditional ML and deep learning-based models respectively.
   11. The best performing model was then evaluated on the unseen data.
   12. the unseen dataset was also 
# Sentiment Prediction Results
